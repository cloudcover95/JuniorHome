# path: src/memory/graph_memory_blackbox.py

"""
GraphMemoryBlackbox

Deepened support for:
- Structured Tasks + Typed Deliverables with status tracking
- Lightweight pub/sub style notifications (local callbacks + event log)
- Strong provenance linking back to BitNet/ternary operations and plasticity

Enables clean, auditable handoffs between components in the BitNet layer.
"""

from typing import Any, Callable, Dict, List, Optional, Set
import time
import hashlib

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

        # Deliverable & Task tracking
        self._deliverables_by_type: Dict[str, Set[str]] = {}
        self._deliverables_by_producer: Dict[str, Set[str]] = {}
        self._tasks: Dict[str, Dict[str, Any]] = {}  # task_id -> task data

        # Lightweight notification system
        self._subscribers: List[Callable] = []
        self._event_log: List[Dict[str, Any]] = []

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

    def store_pattern(self, pattern: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> str:
        node_id = self._make_node_id(pattern)

        if node_id in self.nodes:
            if node_id not in self.temporal_history:
                self.temporal_history[node_id] = []
            self.temporal_history[node_id].append({"timestamp": time.time(), "data": pattern})
            return node_id

        self.nodes[node_id] = {
            "data": pattern,
            "metadata": metadata or {},
            "stored_at": time.time(),
            "node_id": node_id
        }

        if self.hybrid_quantizer:
            raw_vec = self._to_ternary_vector(pattern)
            self.embeddings[node_id] = self.hybrid_quantizer.quantize(raw_vec).tolist()
        elif self.sensitivity_optimizer:
            self.embeddings[node_id] = self.sensitivity_optimizer.quantize_to_ternary(self._to_ternary_vector(pattern)).tolist()
        else:
            self.embeddings[node_id] = self._to_ternary_vector(pattern)

        if self._fast_index is not None:
            tag = pattern.get("type", "general")
            if tag not in self._fast_index:
                self._fast_index[tag] = set()
            self._fast_index[tag].add(node_id)

        for existing_id, emb in list(self.embeddings.items()):
            if existing_id == node_id:
                continue
            if self._cosine_similarity(self.embeddings[node_id], emb) > 0.65:
                if node_id not in self.edges:
                    self.edges[node_id] = set()
                self.edges[node_id].add(existing_id)
                if existing_id not in self.edges:
                    self.edges[existing_id] = set()
                self.edges[existing_id].add(node_id)

        if node_id not in self.temporal_history:
            self.temporal_history[node_id] = []
        self.temporal_history[node_id].append({"timestamp": time.time(), "data": pattern})

        return node_id

    # === Structured Tasks + Typed Deliverables ===
    def post_task(self, task_data: Dict[str, Any], assigned_to: Optional[str] = None, priority: int = 0) -> str:
        """Post a structured task that can be consumed by other components."""
        task_id = self._make_node_id(task_data)
        task = {
            "task_id": task_id,
            "data": task_data,
            "status": "pending",
            "assigned_to": assigned_to,
            "priority": priority,
            "created_at": time.time(),
            "updated_at": time.time()
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
        self._log_event("task_updated", self._tasks[task_id])
        return True

    def get_pending_tasks(self, assigned_to: Optional[str] = None) -> List[Dict[str, Any]]:
        return [t for t in self._tasks.values() if t["status"] == "pending" and (assigned_to is None or t.get("assigned_to") == assigned_to)]

    def post_deliverable(self, data: Dict[str, Any], produced_by: str, deliverable_type: str = "general", version: int = 1, plasticity_signal: Optional[Dict] = None) -> str:
        """
        Post a typed deliverable with strong provenance.
        Can optionally attach plasticity training signals.
        """
        metadata = {
            "produced_by": produced_by,
            "deliverable_type": deliverable_type,
            "version": version,
            "timestamp": time.time(),
            "provenance": f"{produced_by}:{deliverable_type}:v{version}",
            "plasticity_signal_id": plasticity_signal.get("signal_id") if plasticity_signal else None
        }
        node_id = self.store_pattern(data, metadata=metadata)

        if deliverable_type not in self._deliverables_by_type:
            self._deliverables_by_type[deliverable_type] = set()
        self._deliverables_by_type[deliverable_type].add(node_id)

        if produced_by not in self._deliverables_by_producer:
            self._deliverables_by_producer[produced_by] = set()
        self._deliverables_by_producer[produced_by].add(node_id)

        self._log_event("deliverable_posted", {"node_id": node_id, "produced_by": produced_by, "type": deliverable_type})
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

    # === Lightweight Pub/Sub Notifications ===
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
        # Notify subscribers
        for callback in self._subscribers:
            try:
                callback(event)
            except Exception as e:
                print(f"[GraphMemoryBlackbox] Notification error: {e}")

    def get_recent_events(self, limit: int = 20) -> List[Dict[str, Any]]:
        return self._event_log[-limit:]

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

    # === Utility methods ===
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
        print("[GraphMemoryBlackbox] Running deepened deliverable system...")
        task_id = self.post_task({"action": "generate_design"}, assigned_to="VLMDesignAgent")
        design_id = self.post_deliverable({"wing_sweep": 48}, produced_by="VLMDesignAgent", deliverable_type="design")
        self.update_task_status(task_id, "completed", result={"design_id": design_id})
        artifacts_id = self.post_deliverable({"files": ["design.step"]}, produced_by="CADScriptGenerator", deliverable_type="artifact")
        events = self.get_recent_events(5)
        success = len(events) >= 2
        print(f"[GraphMemoryBlackbox] Self-test {'PASSED' if success else 'FAILED'}")
        return success


if __name__ == "__main__":
    gmb = GraphMemoryBlackbox(enable_ternary=True, fast_lookup=True)
    gmb.run_self_test()
