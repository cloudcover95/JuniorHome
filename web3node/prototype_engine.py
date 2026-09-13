"""JuniorLLM prototype-engine sidecar. Does not replace the numpy clock."""
from __future__ import annotations
from fieldcore_bridge import pick_juniorllm_port
from home_terraform import inject
from ternary_kernel import kernel_list, pick_path

def engine(ask="prototype field rails"):
    tf = inject(ask)
    return {"sidecar": "JuniorLLM", "role": "prototype-engine", "replaces_numpy": False,
            "clock": pick_path(), "port": pick_juniorllm_port(ask),
            "terraform": {k: tf.get(k) for k in ("port", "ok", "fusion_backend")},
            "kernel": kernel_list([0.2, -0.4, 0.1, 0.0], [0.3, -0.1, 0.2, 0.05])}

if __name__ == "__main__":
    import json
    print(json.dumps(engine(), indent=2))
