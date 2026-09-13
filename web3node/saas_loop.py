"""Local open SaaS loop."""
from __future__ import annotations
import json
from junioros_prod import main as prod
from latch import latch
from quant_error import check
from tritpack import workflow

def live():
    pack, q, home, node = workflow(), check(), prod(), latch()
    return {"live": True, "saas": "local-open", "jtr1": pack.get("file"),
            "llama_cpp": False, "quant_ok": q.get("ok"), "prod_ok": home.get("suite_ok"),
            "latch": node.get("beta"), "port": pack.get("port")}

if __name__ == "__main__":
    print(json.dumps(live(), indent=2, default=str))
