"""Business crypto address check and agent fulfillment. Local only."""
from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
from typing import Any
from sqlmodel import Session, create_engine
from models.models import Order
from models.comms import advance
from models.payments import write_invoice, crypto_receipt

DB = Path.home() / ".juniorhome" / "juniorstack.db"
ADDR_PATH = Path.home() / ".juniorhome" / "juniorstack" / "crypto_addresses.json"
FULFILL_LOG = Path.home() / ".juniorhome" / "os" / "fulfillment.jsonl"

def load_addresses() -> dict[str, str]:
    if not ADDR_PATH.exists():
        return {}
    return json.loads(ADDR_PATH.read_text(encoding="utf-8"))

def save_addresses(addrs: dict[str, str]) -> Path:
    ADDR_PATH.parent.mkdir(parents=True, exist_ok=True)
    ADDR_PATH.write_text(json.dumps(addrs, indent=2), encoding="utf-8")
    return ADDR_PATH

def check_addresses() -> dict[str, Any]:
    addrs = load_addresses()
    body = {"addresses": addrs, "count": len(addrs), "local": True, "chain_lookup": False, "ts": datetime.utcnow().isoformat()}
    out = Path.home() / ".juniorhome" / "os" / "crypto_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return body

def agent_fulfill(order_id: int, method: str = "local_trit") -> dict[str, Any]:
    check = check_addresses()
    if not check["addresses"]:
        return {"ok": False, "reason": "no business addresses configured", "order_id": order_id}
    engine = create_engine(f"sqlite:///{DB}")
    with Session(engine) as session:
        order = session.get(Order, order_id)
        if not order:
            return {"ok": False, "reason": "no order"}
        if order.status not in ("received", "making"):
            return {"ok": False, "reason": f"status {order.status} not eligible"}
    receipt = crypto_receipt(order_id, method)
    write_invoice(order_id)
    result = advance(order_id, "making")
    log = {"order_id": order_id, "method": method, "addresses": list(check["addresses"].keys()), "receipt": receipt.get("invoice_hash"), "advanced": result.get("ok"), "ts": datetime.utcnow().isoformat(), "local": True}
    FULFILL_LOG.parent.mkdir(parents=True, exist_ok=True)
    with FULFILL_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(log) + "\n")
    return {"ok": True, "order_id": order_id, "status": "making", "receipt": receipt}
