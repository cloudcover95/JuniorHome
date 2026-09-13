"""Modular adapters: collect → reduce → memory → route → train."""
from __future__ import annotations
from pathlib import Path
from agent_stack import run as agents
from barebones import tick
from fieldcore_bridge import pick_juniorllm_port
from fleet import isolate
from second_brain import write_note

def collect(n=3):
    ticks = [tick([0.2 + i * 0.05, -0.1, 0.3], [0.4, 0.0, -0.2], 12.0) for i in range(n)]
    return {"stage": "collect", "tickets": ticks}

def reduce(payload):
    ticks = payload.get("tickets") or []
    n = len(ticks) or 1
    return {**payload, "stage": "reduce", "mean_y": sum(t["y"] for t in ticks) / n,
            "mean_gamma": sum(t["gamma"] for t in ticks) / n, "n": n}

def memory(payload):
    note = write_note(Path(__file__).resolve().parent / "vault",
                      f"# adapter memory\nmean_y={payload.get('mean_y')}\n")
    return {**payload, "stage": "memory", "note": str(note)}

def route(payload):
    ask = "juniorosai field adapter pipe"
    ag = agents(ask)
    return {**payload, "stage": "route", "port": pick_juniorllm_port(ask),
            "agents_ok": ag.get("ok"), "patterns": ag.get("patterns")}

def train(payload, run=False):
    return {**payload, "stage": "train", "run": run,
            "spark": isolate({"workload": "ue5 spark fine-tune"}, 90), "ue_boot": bool(run)}

def run(n=3, train_run=False):
    row = train(route(memory(reduce(collect(n)))), run=train_run)
    row["framework"] = "JuniorOSai-adapters"
    row["not"] = "zeRO-3"
    return row

if __name__ == "__main__":
    import json
    out = run()
    print(json.dumps({k: out[k] for k in ("framework", "n", "mean_y", "port", "run", "not")}, indent=2))
