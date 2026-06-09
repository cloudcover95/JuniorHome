# path: src/memory/graph_memory_blackbox.py

"""
GraphMemoryBlackbox

Workflow orchestration and reactive subscription enhancements.
"""

from typing import Any, Callable, Dict, List, Optional, Set
import time
import hashlib
import os

try:
    from src.quantization.hybrid_squeeze_bitnet import HybridSqueezeBitNetQuantizer
    from src.quantization.sensitivity_ternary_optimizer import SensitivityTernaryOptimizer
except ImportError:
    HybridSqueezeBitNetQuantizer = None
    SensitivityTernaryOptimizer = None


class GraphMemoryBlackbox:
    def __init__(self, node_id: str = "default", enable_ternary: bool = True, embedding_fn: Optional[Callable] = None, fast_lookup: bool = True, sensitivity_optimizer: Optional[Any] = None):
        self.node_id = node_id
        self.enable_ternary = enable_ternary
        self.embedding_fn = embedding_fn
        self.fast_lookup = fast_lookup
        self.sensitivity_optimizer = sensitivity_optimizer
        self.hybrid_quantizer = HybridSqueezeBitNetQuantizer() if HybridSqueezeBitNetQuantizer else None

        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.edges: Dict[str, Set[str]] = {}
        self.embeddings: Dict[str, List[float]] = {}
        self.temporal_history: Dict[str, List[Dict[str, Any]]] = {}
        self._fast_index: Dict[str, Set[str]] = {} if fast_lookup else None

        self._deliverables_by_type: Dict[str, Set[str]] = {}
        self._deliverables_by_producer: Dict[str, Set[str]] = {}
        self._tasks: Dict[str, Dict[str, Any]] = {}

        self._subscribers: List[Callable] = []
        self._event_log: List[Dict[str, Any]] = []
        self._type_subscribers: Dict[str, List[Callable]] = {}

    def _make_node_id(self, data: Dict[str, Any]) -> str:
        serialized = str(sorted(data.items())).encode()
        return hashlib.sha256(serialized).hexdigest()[:16]

    def _to_ternary_vector(self, data: Dict[str, Any]) -> List[float]:
        if self.embedding_fn:
            try:
                raw_embedding = self.embedding_fn(data)
                if self.hybrid_quantizer:
                    return self.hybrid_quantizer.quantize(raw_embedding).tolist()
                if self.sensitivity_optimizer:
                    return self.sensitivity_optimizer.quantize_to_ternary(raw_embedding).tolist()
                return raw_embedding
            except Exception as e:
                print(f"[GraphMemoryBlackbox] Embedding error: {e}")

        if not self.enable_ternary:
            return [0.0] * 8

        vec = []
        for key, val in sorted(data.items()):
            if isinstance(val, (int, float)):
                v = float(val)
                vec.append(1.0 if v > 0.5 else (-1.0 if v < -0.5 else 0.0))
            else:
                vec.append(0.0)
        while len(vec) < 8:
            vec.append(0.0)
        return vec[:8]

    def export_to_obsidian_vault(self, vault_path: str, node_ids: Optional[List[str]] = None, include_plasticity: bool = True) -> int:
        os.makedirs(vault_path, exist_ok=True)
        exported = 0
        targets = node_ids or list(self.nodes.keys())[-50:]

        for node_id in targets:
            if node_id not in self.nodes:
                continue
            node = self.nodes[node_id]
            meta = node.get("metadata", {})
            data = node.get("data", {})

            filename = f"{meta.get('produced_by', 'unknown')}_{meta.get('deliverable_type', 'item')}_{node_id[:8]}.md"
            filepath = os.path.join(vault_path, filename)

            content = f"# {meta.get('deliverable_type', 'Item').title()} — {meta.get('produced_by', 'Unknown')}\n\n"
            content += f"**Node ID**: `{node_id}`\n"
            content += f"**Timestamp**: {meta.get('timestamp')}\n"
            content += f"**Version**: {meta.get('version', 1)}\n\n"

            content += "## Provenance\n"
            content += f"- Produced by: **{meta.get('produced_by')}**\n"
            content += f"- BitNet operation: `{meta.get('bitnet_operation', 'N/A')}`\n"
            content += f"- Plasticity signal: `{meta.get('plasticity_signal_id', 'N/A')}`\n\n"

            if include_plasticity and meta.get('plasticity_signal_id'):
                content += "## Plasticity Signal\n```json\n"
                content += str(meta)[:1500]
                content += "\n```\n\n"

            content += "## Data\n```json\n"
            content += str(data)[:2500]
            content += "\n```\n"

            try:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(content)
                exported += 1
            except Exception as e:
                print(f"[GraphMemoryBlackbox] Obsidian export error for {node_id}: {e}")

        return exported

    def export_for_obsidian(self, node_id: str, include_provenance: bool = True) -> str:
        if node_id not in self.nodes:
            return "# Not found"

        node = self.nodes[node_id]
        meta = node.get("metadata", {})
        data = node.get("data", {})

        md = f"# {meta.get('deliverable_type', 'Item')} - {meta.get('produced_by', 'Unknown')}\n\n"
        md += f"**Timestamp**: {meta.get('timestamp', 'N/A')}\n        md += f"**Version**: {meta.get('version', 1)}\n\n"

        if include_provenance:
            md += "## Provenance\n"
            md += f"- Produced by: {meta.get('produced_by')}\n"
            md += f"- BitNet operation: {meta.get('bitnet_operation', 'N/A')}\n"
            md += f"- Plasticity signal: {meta.get('plasticity_signal_id', 'N/A')}\n\n"

        md += "## Data\n```json\n"
        md += str(data)[:2000]
        md += "\n```\n"

        return md

    def post_task(self, task_data: Dict[str, Any], assigned_to: Optional[str] = None, priority: int = 0, depends_on: Optional[List[str]] = None) -> str:
        task_id = self._make_node_id(task_data)
        task = {
            "task_id": task_id,
            "data": task_data,
            "status": "pending",
            "assigned_to": assigned_to,
            "priority": priority,
            "depends_on": depends_on or [],
            "created_at": time.time(),
            "updated_at": time.time(),
            "result_node_id": None
        }
        self._tasks[task_id] = task
        self._log_event("task_posted", task)
        return task_id

    def update_task_status(self, task_id: str, status: str, result: Optional[Dict[str, Any]] = None) -> bool:
        if task_id not in self._tasks:
            return False
        self._tasks[task_id]["status"] = status
        self._tasks[task_id]["updated_at"] = time.time()
        if result:
            self._tasks[task_id]["result"] = result
            if "node_id" in result:
                self._tasks[task_id]["result_node_id"] = result["node_id"]
        self._log_event("task_updated", self._tasks[task_id])
        return True

    def claim_task(self, task_id: str, claimed_by: str) -> bool:
        if task_id not in self._tasks:
            return False
        task = self._tasks[task_id]
        if task["status"] != "pending":
            return False
        for dep_id in task.get("depends_on", []):
            if dep_id in self._tasks and self._tasks[dep_id]["status"] != "completed":
                return False
        task["status"] = "in_progress"
        task["assigned_to"] = claimed_by
        task["updated_at"] = time.time()
        self._log_event("task_claimed", task)
        return True

    def get_pending_tasks(self, assigned_to: Optional[str] = None) -> List[Dict[str, Any]]:
        return [t for t in self._tasks.values() if t["status"] == "pending" and (assigned_to is None or t.get("assigned_to") == assigned_to)]

    def get_ready_tasks(self, assigned_to: Optional[str] = None) -> List[Dict[str, Any]]:
        ready = []
        for t in self._tasks.values():
            if t["status"] != "pending":
                continue
            if assigned_to and t.get("assigned_to") != assigned_to:
                continue
            deps_met = True
            for dep_id in t.get("depends_on", []):
                if dep_id in self._tasks and self._tasks[dep_id]["status"] != "completed":
                    deps_met = False
                    break
            if deps_met:
                ready.append(t)
        return ready

    def post_deliverable(self, data: Dict[str, Any], produced_by: str, deliverable_type: str = "general", version: int = 1, plasticity_signal: Optional[Dict] = None, bitnet_metadata: Optional[Dict] = None) -> str:
        metadata = {
            "produced_by": produced_by,
            "deliverable_type": deliverable_type,
            "version": version,
            "timestamp": time.time(),
            "provenance": f"{produced_by}:{deliverable_type}:v{version}",
            "plasticity_signal_id": plasticity_signal.get("signal_id") if plasticity_signal else None,
            "bitnet_operation": bitnet_metadata.get("operation") if bitnet_metadata else None
        }
        node_id = self.store_pattern(data, metadata=metadata)

        if deliverable_type not in self._deliverables_by_type:
            self._deliverables_by_type[deliverable_type] = set()
        self._deliverables_by_type[deliverable_type].add(node_id)

        if produced_by not in self._deliverables_by_producer:
            self._deliverables_by_producer[produced_by] = set()
        self._deliverables_by_producer[produced_by].add(node_id)

        self._log_event("deliverable_posted", {
            "node_id": node_id,
            "produced_by": produced_by,
            "type": deliverable_type
        })
        if deliverable_type in self._type_subscribers:
            for cb in self._type_subscribers[deliverable_type]:
                try:
                    cb({"type": "deliverable_posted", "node_id": node_id, "produced_by": produced_by})
                except Exception as e:
                    print(f"[GraphMemoryBlackbox] Type subscriber error: {e}")
        return node_id

    def get_deliverables(self, deliverable_type: Optional[str] = None, produced_by: Optional[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
        results = []
        candidates = set(self.nodes.keys())

        if deliverable_type and deliverable_type in self._deliverables_by_type:
            candidates = self._deliverables_by_type[deliverable_type]
        elif produced_by and produced_by in self._deliverables_by_producer:
            candidates = self._deliverables_by_producer[produced_by]

        for node_id in list(candidates)[:limit]:
            if node_id in self.nodes:
                results.append(self.nodes[node_id])

        results.sort(key=lambda x: x.get("metadata", {}).get("timestamp", 0), reverse=True)
        return results

    def get_latest_deliverable(self, deliverable_type: str, produced_by: Optional[str] = None) -> Optional[Dict[str, Any]]:
        results = self.get_deliverables(deliverable_type=deliverable_type, produced_by=produced_by, limit=1)
        return results[0] if results else None

    def get_provenance_chain(self, node_id: str, max_depth: int = 8) -> List[Dict[str, Any]]:
        chain = []
        visited = set()
        current_id = node_id
        depth = 0

        while current_id and current_id not in visited and depth < max_depth:
            visited.add(current_id)
            if current_id not in self.nodes:
                break
            node = self.nodes[current_id]
            chain.append(node)

            meta = node.get("metadata", {})
            provenance = meta.get("provenance", "")
            if ":" in provenance:
                parent_producer = provenance.split(":")[0]
                for nid, n in self.nodes.items():
                    if n.get("metadata", {}).get("produced_by") == parent_producer and nid != current_id:
                        current_id = nid
                        break
                else:
                    break
            else:
                break
            depth += 1

        return chain

    def get_artifacts_from_design(self, design_node_id: str) -> List[Dict[str, Any]]:
        results = []
        for node in self.nodes.values():
            meta = node.get("metadata", {})
            if meta.get("deliverable_type") == "artifact":
                if design_node_id in str(meta.get("provenance", "")):
                    results.append(node)
        return results

    def get_next_work_items(self, component: str = None) -> Dict[str, List[Dict[str, Any]]]:
        ready_tasks = self.get_ready_tasks(assigned_to=component)
        latest_designs = self.get_deliverables(deliverable_type="design", limit=5)
        latest_artifacts = self.get_deliverables(deliverable_type="artifact", limit=5)
        return {
            "ready_tasks": ready_tasks,
            "latest_designs": latest_designs,
            "latest_artifacts": latest_artifacts
        }

    def subscribe_to_deliverable_type(self, deliverable_type: str, callback: Callable):
        if deliverable_type not in self._type_subscribers:
            self._type_subscribers[deliverable_type] = []
        if callback not in self._type_subscribers[deliverable_type]:
            self._type_subscribers[deliverable_type].append(callback)

    def subscribe(self, callback: Callable):
        if callback not in self._subscribers:
            self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable):
        if callback in self._subscribers:
            self._subscribers.remove(callback)

    def _log_event(self, event_type: str, payload: Dict[str, Any]):
        event = {
            "event_type": event_type,
            "payload": payload,
            "timestamp": time.time()
        }
        self._event_log.append(event)
        for callback in self._subscribers:
            try:
                callback(event)
            except Exception as e:
                print(f"[GraphMemoryBlackbox] Notification error: {e}")

    def get_recent_events(self, event_type: Optional[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
        events = self._event_log
        if event_type:
            events = [e for e in events if e["event_type"] == event_type]
        return events[-limit:]

    def notify_deliverable_ready(self, deliverable_type: str, produced_by: str) -> Dict[str, Any]:
        latest = self.get_latest_deliverable(deliverable_type, produced_by)
        event = {
            "deliverable_type": deliverable_type,
            "produced_by": produced_by,
            "ready": latest is not None,
            "node_id": latest.get("node_id") if latest else None,
            "timestamp": time.time()
        }
        self._log_event("deliverable_ready", event)
        return event

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5 or 1e-8
        norm_b = sum(x * x for x in b) ** 0.5 or 1e-8
        return dot / (norm_a * norm_b)

    def query_similar(self, query_pattern: Dict[str, Any], top_k: int = 5) -> List[Dict[str, Any]]:
        if self._fast_index is not None:
            tag = query_pattern.get("type", "general")
            candidates = self._fast_index.get(tag, set())
            if candidates:
                scored = []
                query_vec = self._to_ternary_vector(query_pattern)
                for node_id in list(candidates)[:300]:
                    if node_id in self.embeddings:
                        sim = self._cosine_similarity(query_vec, self.embeddings[node_id])
                        scored.append((sim, self.nodes.get(node_id, {})))
                scored.sort(reverse=True, key=lambda x: x[0])
                return [item[1] for item in scored[:top_k] if item[1]]

        query_vec = self._to_ternary_vector(query_pattern)
        scored = []
        for node_id, emb in self.embeddings.items():
            sim = self._cosine_similarity(query_vec, emb)
            scored.append((sim, self.nodes.get(node_id, {})))
        scored.sort(reverse=True, key=lambda x: x[0])
        return [item[1] for item in scored[:top_k] if item[1]]

    def store_spike_event(self, spike_data: Dict[str, Any]):
        spike_data["is_spike_event"] = True
        return self.store_pattern(spike_data, metadata={"event_type": "spike"})

    def export_state(self, include_temporal: bool = True) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "nodes": self.nodes,
            "edges": {k: list(v) for k, v in self.edges.items()},
            "embeddings": self.embeddings,
            "temporal_history": self.temporal_history if include_temporal else {},
            "exported_at": time.time()
        }

    def import_state(self, state: Dict[str, Any]):
        self.node_id = state.get("node_id", self.node_id)
        self.nodes = state.get("nodes", {})
        self.edges = {k: set(v) for k, v in state.get("edges", {}).items()}
        self.embeddings = state.get("embeddings", {})
        self.temporal_history = state.get("temporal_history", {})

    def set_embedding_function(self, fn: Callable):
        self.embedding_fn = fn

    def set_sensitivity_optimizer(self, optimizer: Any):
        self.sensitivity_optimizer = optimizer

    def run_self_test(self) -> bool:
        print("[GraphMemoryBlackbox] Running with enhanced workflow...")
        task_id = self.post_task({"action": "generate_design"}, assigned_to="VLMDesignAgent")
        self.claim_task(task_id, "VLMDesignAgent")
        design_id = self.post_deliverable({"wing_sweep": 48}, produced_by="VLMDesignAgent", deliverable_type="design")
        self.update_task_status(task_id, "completed", result={"node_id": design_id})
        count = self.export_to_obsidian_vault("/tmp/test_obsidian_vault", node_ids=[design_id])
        success = count >= 1
        print(f"[GraphMemoryBlackbox] Self-test {'PASSED' if success else 'FAILED'}")
        return success


if __name__ == "__main__":
    gmb = GraphMemoryBlackbox(enable_ternary=True, fast_lookup=True)
    gmb.run_self_test()
