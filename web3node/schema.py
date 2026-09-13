"""JSONL documents. Mongo later optional."""
from __future__ import annotations
import json, time
from pathlib import Path

def ticket_doc(y, gamma, n, node):
    return {"_col": "tickets", "t": int(time.time()*1000), "quant": "ternary-1.58",
            "y": y, "gamma": gamma, "n": n, "node": node}

def append(doc, vault=None):
    vault = vault or Path(__file__).resolve().parent / "vault" / "mongo_shape.jsonl"
    vault.parent.mkdir(parents=True, exist_ok=True)
    with vault.open("a", encoding="utf-8") as h:
        h.write(json.dumps(doc) + "\n")
    return vault

INDEXES = {"tickets": (("node", 1), ("t", -1))}
