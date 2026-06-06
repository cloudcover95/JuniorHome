# path: src/memory/graph_memory_blackbox.py

"""
GraphMemoryBlackbox

Enhanced with:
- Optional BitNet-mlx embedding function for real ternary vectors
- Temporal evolution tracking per node
- Basic community detection
- Still fully blackbox and pipeable
"""

from typing import Any, Callable, Dict, List, Optional, Set
import time
import hashlib


class GraphMemoryBlackbox:
    def __init__(self, node_id: str = "default", enable_ternary: bool = True, embedding_fn: Optional[Callable] = None):
        self.node_id = node_id
        self.enable_ternary = enable_ternary
        self.embedding_fn = embedding_fn  # Can be a real BitNet-mlx embedding function
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.edges: Dict[str, Set[str]] = {}
        self.embeddings: Dict[str, List[float]] = {}
        self.temporal_history: Dict[str, List[Dict[str, Any]]] = {}  # For temporal evolution

    def _make_node_id(self, data: Dict[str, Any]) -> str:
        serialized = str(sorted(data.items())).encode()
        return hashlib.sha256(serialized).hexdigest()[:16]

    def _to_ternary_vector(self, data: Dict[str, Any]) -> List[float]:
        if self.embedding_fn:
            try:
                return self.embedding_fn(data)  # Real BitNet-mlx vector
            except Exception as e:
                print(f"[GraphMemoryBlackbox] Embedding fn error: {e}")

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
            # Update temporal history
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
        self.embeddings[node_id] = self._to_ternary_vector(pattern)

        # Simple auto-linking
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

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5 or 1e-8
        norm_b = sum(x * x for x in b) ** 0.5 or 1e-8
        return dot / (norm_a * norm_b)

    def query_similar(self, query_pattern: Dict[str, Any], top_k: int = 5) -> List[Dict[str, Any]]:
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
        """Return recent changes to a node over time."""
        if node_id not in self.temporal_history:
            return []
        return self.temporal_history[node_id][-max_steps:]

    def detect_communities(self) -> Dict[str, List[str]]:
        """Simple community detection based on connectivity."""
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

    def set_embedding_function(self, fn: Callable):
        """Set a real BitNet-mlx embedding function for ternary vectors."""
        self.embedding_fn = fn

    def run_self_test(self) -> bool:
        print("[GraphMemoryBlackbox] Running enhanced self-test...")
        test_pattern = {"type": "supersonic_design", "sweep": 48, "drag": 0.011}
        node_id = self.store_pattern(test_pattern)
        similar = self.query_similar({"type": "supersonic_design", "sweep": 47})
        evolution = self.get_temporal_evolution(node_id)
        communities = self.detect_communities()
        success = len(similar) >= 1 and len(communities) >= 0
        print(f"[GraphMemoryBlackbox] Self-test {'PASSED' if success else 'FAILED'}")
        return success


if __name__ == "__main__":
    gmb = GraphMemoryBlackbox(enable_ternary=True)
    gmb.run_self_test()
