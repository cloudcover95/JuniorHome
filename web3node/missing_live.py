"""Backlog → runnable. Flags stay honest."""
import json
from adapters import run as adapters_run
from community_saas import credit, token
from gguf_i2s import write_i2s
from latch import latch
from quant_error import check
from ste_tape import step
from tritpack import workflow
from ue_mini import mini

def live():
    pack, gg = workflow(), write_i2s([0.2, -0.4, 0.1, 0.0]*8)
    tape = step([0.3, -0.2, 0.1], [1.0, 0.5, -0.2], 0.0)
    return {"i2s_gguf": gg, "ste": {k: tape[k] for k in ("y", "err", "ste")},
            "int4_check": check().get("ok"), "mongo": False, "google_oauth": False,
            "community_token": token("cu-aurora"), "ledger": credit("cu-aurora", 1.0, "missing-live").get("hash"),
            "adapters_port": adapters_run(n=3).get("port"), "ue_mini": mini(),
            "saas": "local-community-ledger", "jtr1": pack.get("file"), "latch": latch().get("beta")}

if __name__ == "__main__":
    print(json.dumps(live(), indent=2, default=str))
