# path: src/memory/graph_memory_blackbox.py

"""
GraphMemoryBlackbox

Added basic spiking/event integration hooks.
Patterns can now be tagged as spike events for neuromorphic workflows.
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

    def store_spike_event(self, spike_data: Dict[str, Any]):
        """Store a pattern as a spike event for neuromorphic workflows."""
        spike_data["is_spike_event"] = True
        return self.store_pattern(spike_data, metadata={"event_type": "spike"})

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

    def infer_relations(self, node_id: str, depth: int = 2) -> List[Dict[str, Any]]:
        if node_id not in self.edges:
            return []
        results = []
        visited = set()
        queue = [(node_id, 0)]
        while queue:
            current, d = queue.pop(0)
            if current in visited or d > depth:
                continue
            visited.add(current)
            if current in self.nodes:
                results.append(self.nodes[current])
            for neighbor in self.edges.get(current, []):
                if neighbor not in visited:
                    queue.append((neighbor, d + 1))
        return results

    def get_temporal_evolution(self, node_id: str, max_steps: int = 5) -> List[Dict[str, Any]]:
        if node_id not in self.temporal_history:
            return []
        return self.temporal_history[node_id][-max_steps:]

    def detect_communities(self) -> Dict[str, List[str]]:
        communities = {}
        visited = set()
        community_id = 0
        for node in list(self.nodes.keys()):
            if node in visited:
                continue
            community = []
            stack = [node]
            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)
                community.append(current)
                for neighbor in self.edges.get(current, []):
                    if neighbor not in visited:
                        stack.append(neighbor)
            communities[f"community_{community_id}"] = community
            community_id += 1
        return communities

    def get_context_for_agent(self, query: Dict[str, Any], max_context: int = 3) -> Dict[str, Any]:
        similar = self.query_similar(query, top_k=max_context)
        return {
            "similar_patterns": [s.get("data", {}) for s in similar],
            "count": len(similar)
        }

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
        print("[GraphMemoryBlackbox] Running with spiking hooks...")
        test_pattern = {"type": "spike_event", "sweep": 48}
        node_id = self.store_spike_event(test_pattern)
        similar = self.query_similar({"type": "spike_event"})
        success = len(similar) >= 1
        print(f"[GraphMemoryBlackbox] Self-test {'PASSED' if success else 'FAILED'}")
        return success


if __name__ == "__main__":
    gmb = GraphMemoryBlackbox(enable_ternary=True, fast_lookup=True)
    gmb.run_self_test()
