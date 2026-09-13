"""T4 tickets → T0 brain → agents. Spark staged."""
from __future__ import annotations
import json
from pathlib import Path
from agent_stack import run as agents
from barebones import tick
from compute_layers import expand
from fleet import isolate
from second_brain import write_note

def allreduce(tickets):
    ys = [t["y"] for t in tickets]
    gs = [t["gamma"] for t in tickets]
    n = len(ys) or 1
    return {"mean_y": sum(ys) / n, "mean_gamma": sum(gs) / n, "n": n}

def pipeline(nodes=3):
    edge = [tick([0.2 + i * 0.05, -0.1, 0.3], [0.4, 0.0, -0.2], 12.0) for i in range(nodes)]
    reduced = allreduce(edge)
    layers = expand("dist pipe infer train")
    routed = agents("juniorosai field dist pipe flagstaff")
    note = write_note(Path(__file__).resolve().parent / "vault",
                      f"# dist pipe\nnodes={nodes} mean_y={reduced['mean_y']:.4f}\n")
    spark = isolate({"workload": "ue5 spark fine-tune"}, 90)
    row = {"kind": "pipeline", "not": "megatron-ddp",
           "edge_ai": {"nodes": nodes, "opt": "trit ticket", "reduce": reduced},
           "home": isolate({"workload": "m6 home infer"}, 45),
           "second_brain": str(note),
           "agents": {"ok": routed.get("ok"), "port": (routed.get("terraform") or {}).get("port"),
                       "patterns": routed.get("patterns")},
           "spark_stage": {**spark, "run": False},
           "layers": {k: v.get("node") for k, v in layers["layers"].items()}}
    out = Path(__file__).resolve().parent / "vault" / "dist_pipe.json"
    out.write_text(json.dumps(row, indent=2, default=str), encoding="utf-8")
    row["out"] = str(out)
    return row

if __name__ == "__main__":
    print(json.dumps(pipeline(), indent=2, default=str))
