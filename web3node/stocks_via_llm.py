"""StocksNode math stays in stocksnode. Routing through JuniorLLM port names."""
from __future__ import annotations
from typing import Any
from fieldcore_bridge import pick_juniorllm_port, stocksnode_manifold
from home_terraform import terraform
from sparse_formats import compare_formats

def infer(task: str = "q_mark field ticker") -> dict[str, Any]:
    port = pick_juniorllm_port(task)
    try:
        import importlib
        live = importlib.import_module("ports.registry").pick(task, 8.0)
        port = getattr(live, "name", port)
    except Exception:
        pass
    features = stocksnode_manifold(n=24, t=40, seed=9)
    return {
        "juniorllm_port": port,
        "terraform_port": terraform(task).get("port"),
        "stocksnode": {
            "q_mark_mean": float(features["q_mark"].mean()),
            "spot_mean": float(features["spot"].mean()),
            "z_mean": float(features["z_score"].mean()),
        },
        "sparse": compare_formats(features["close"]),
        "source": "src.stocksnode.core.financial_tensor when importable",
    }

if __name__ == "__main__":
    import json
    print(json.dumps(infer(), indent=2, default=str))
