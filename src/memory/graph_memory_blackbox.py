# path: src/memory/graph_memory_blackbox.py

"""
GraphMemoryBlackbox

A modular, blackbox-style Graph Memory component designed as BitNet codebase technology.

Features:
- Pluggable interface (can be swapped or piped into RealDataRunner, VLMDesignAgent, PlasticityEngine)
- Ternary-aware node/edge embeddings (aligns with BitNet 1.58/3.0 philosophy)
- Pattern storage, similarity search, and inference
- Simple built-in test harness

This acts as a sovereign, efficient graph memory layer that can integrate with:
- VLMDesignAgent (design pattern memory)
- CallPatternStore / JuniorMemSys (recognition + design graphs)
- Plasticity outcomes (learn which graph nodes are important)
- RealDataRunner (store simulation results as graph events)

Blackbox contract:
- Input: pattern dicts or events
- Output: similar patterns, inferred relations, context vectors
- No hard dependencies on specific LLM or hardware
"""

from typing import Any, Dict, List, Optional, Set
import time
import hashlib


class GraphMemoryBlackbox:
    def __init__(self, node_id: str = "default", enable_ternary: bool = True):
        self.node_id = node_id
        self.enable_ternary = enable_ternary
        self.nodes: Dict[str, Dict[str, Any]] = {}          # node_id -> data
        self.edges: Dict[str, Set[str]] = {}                # node_id -> connected nodes
        self.embeddings: Dict[str, List[float]] = {}        # node_id -> ternary-ish vector

    def _make_node_id(self, data: Dict[str, Any]) -> str:
        serialized = str(sorted(data.items())).encode()
        return hashlib.sha256(serialized).hexdigest()[:16]

    def _to_ternary_vector(self, data: Dict[str, Any]) -> List[float]:
        """Simple ternary projection (placeholder for real BitNet embedding)."""
        if not self.enable_ternary:
            return [0.0] * 8
        # Very lightweight ternary hash-like embedding
        vec = []
        for key, val in sorted(data.items()):
            if isinstance(val, (int, float)):
                v = float(val)
                vec.append(1.0 if v > 0.5 else (-1.0 if v < -0.5 else 0.0))
            else:
                vec.append(0.0)
        # Pad or truncate to fixed size
        while len(vec) < 8:
            vec.append(0.0)
        return vec[:8]

    def store_pattern(self, pattern: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> str:
        """Store a pattern (design, recognition event, simulation result, etc.)."""
        node_id = self._make_node_id(pattern)
        if node_id in self.nodes:
            return node_id

        self.nodes[node_id] = {
            "data": pattern,
            "metadata": metadata or {},
            "stored_at": time.time(),
            "node_id": node_id
        }
        self.embeddings[node_id] = self._to_ternary_vector(pattern)

        # Auto-link to similar existing nodes (simple similarity)
        for existing_id, emb in self.embeddings.items():
            if existing_id == node_id:
                continue
            similarity = self._cosine_similarity(self.embeddings[node_id], emb)
            if similarity > 0.7:
                if node_id not in self.edges:
                    self.edges[node_id] = set()
                self.edges[node_id].add(existing_id)
                if existing_id not in self.edges:
                    self.edges[existing_id] = set()
                self.edges[existing_id].add(node_id)

        return node_id

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        return dot / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 0.0

    def query_similar(self, query_pattern: Dict[str, Any], top_k: int = 5) -> List[Dict[str, Any]]:
        """Return most similar stored patterns."""
        query_vec = self._to_ternary_vector(query_pattern)
        scored = []
        for node_id, emb in self.embeddings.items():
            sim = self._cosine_similarity(query_vec, emb)
            scored.append((sim, self.nodes[node_id]))
        scored.sort(reverse=True, key=lambda x: x[0])
        return [item[1] for item in scored[:top_k]]

    def infer_relations(self, node_id: str, depth: int = 2) -> List[Dict[str, Any]]:
        """Graph inference: find related patterns via edges."""
        if node_id not in self.edges:
            return []
        results = []
        visited = set()
        to_visit = [(node_id, 0)]
        while to_visit:
            current, d = to_visit.pop(0)
            if current in visited or d > depth:
                continue
            visited.add(current)
            if current in self.nodes:
                results.append(self.nodes[current])
            for neighbor in self.edges.get(current, []):
                if neighbor not in visited:
                    to_visit.append((neighbor, d + 1))
        return results

    def get_context_for_agent(self, query: Dict[str, Any], max_context: int = 3) -> Dict[str, Any]:
        """Return compact context suitable for feeding into VLMDesignAgent or LLM."""
        similar = self.query_similar(query, top_k=max_context)
        return {
            "similar_patterns": [s["data"] for s in similar],
            "count": len(similar)
        }

    # --- Simple built-in test harness ---
    def run_self_test(self) -> bool:
        """Basic self-test for the blackbox."""
        print("[GraphMemoryBlackbox] Running self-test...")
        test_pattern = {"type": "supersonic_wing", "sweep": 45, "drag": 0.012}
        node_id = self.store_pattern(test_pattern)
        similar = self.query_similar({"type": "supersonic_wing", "sweep": 44})
        relations = self.infer_relations(node_id)
        context = self.get_context_for_agent({"type": "supersonic_wing"})

        success = (
            len(similar) >= 1 and
            len(context["similar_patterns"]) >= 1
        )
        print(f"[GraphMemoryBlackbox] Self-test {'PASSED' if success else 'FAILED'}")
        return success


if __name__ == "__main__":
    gmb = GraphMemoryBlackbox(enable_ternary=True)
    gmb.run_self_test()
