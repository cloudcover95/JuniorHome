"""Print jobs, eco stickers, RFID tags, local Juniors mint. Local only."""
from __future__ import annotations
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any
from models.trit import pack_status

PRINT_DIR = Path.home() / ".juniorhome" / "juniorstack" / "prints"
TAG_DIR = Path.home() / ".juniorhome" / "juniorstack" / "tags"
MINT_LOG = Path.home() / ".juniorhome" / "os" / "juniors_mint.jsonl"

def _ensure():
    PRINT_DIR.mkdir(parents=True, exist_ok=True)
    TAG_DIR.mkdir(parents=True, exist_ok=True)
    MINT_LOG.parent.mkdir(parents=True, exist_ok=True)

def print_job(order_id: int, sku: str, qty: int, name: str) -> Path:
    _ensure()
    body = {"order_id": order_id, "sku": sku, "name": name, "qty": qty, "ts": datetime.utcnow().isoformat(), "pack": pack_status(f"{sku}:{order_id}"), "local": True}
    path = PRINT_DIR / f"print_{order_id}_{sku}.json"
    path.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return path

def eco_sticker(sku: str, claim: str = "recycled") -> dict[str, Any]:
    body = {"sku": sku, "claim": claim, "material": "paper", "ink": "soy", "pack": pack_status(f"eco:{sku}"), "local": True}
    path = TAG_DIR / f"eco_{sku}.json"
    path.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return body

def rfid_tag(sku: str, item_id: str) -> dict[str, Any]:
    tag = hashlib.sha256(f"{sku}:{item_id}".encode()).hexdigest()[:16]
    body = {"sku": sku, "item_id": item_id, "tag": tag, "pack": pack_status(tag), "local": True}
    path = TAG_DIR / f"rfid_{sku}_{item_id}.json"
    path.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return body

def juniors_mint(order_id: int, sku: str, qty: int) -> dict[str, Any]:
    _ensure()
    body = {"order_id": order_id, "sku": sku, "qty": qty, "mint": "local_juniors", "hash": hashlib.sha256(f"{order_id}:{sku}:{qty}".encode()).hexdigest()[:16], "pack": pack_status(f"mint:{sku}:{order_id}"), "ts": datetime.utcnow().isoformat(), "broadcast": False, "local": True}
    with MINT_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(body) + "\n")
    return body

def on_sale(order_id: int, sku: str, name: str, qty: int, item_id: str = "") -> dict[str, Any]:
    item_id = item_id or f"{order_id}-{sku}"
    return {"print": str(print_job(order_id, sku, qty, name)), "eco": eco_sticker(sku), "rfid": rfid_tag(sku, item_id), "mint": juniors_mint(order_id, sku, qty), "local": True}
