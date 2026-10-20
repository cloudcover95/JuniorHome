"""Payment and invoice rails for JuniorStack. Local only."""
from __future__ import annotations
import json, hashlib
from datetime import datetime
from pathlib import Path
from typing import Any
from sqlmodel import Session, create_engine, select
from models.models import Order, OrderItem, Product

DB = Path.home() / ".juniorhome" / "juniorstack.db"
INVOICE_DIR = Path.home() / ".juniorhome" / "juniorstack" / "invoices"
INVOICE_DIR.mkdir(parents=True, exist_ok=True)

def _engine():
    return create_engine(f"sqlite:///{DB}")

def invoice_body(order_id: int) -> dict[str, Any]:
    engine = _engine()
    with Session(engine) as session:
        order = session.get(Order, order_id)
        if not order:
            return {"ok": False, "reason": "no order"}
        items = session.exec(select(OrderItem).where(OrderItem.order_id == order_id)).all()
        lines = []
        for it in items:
            p = session.get(Product, it.product_id)
            lines.append({"sku": p.sku if p else "?", "name": p.name if p else "?", "qty": it.qty, "price_cents": it.price_cents})
        body = {"order_id": order.id, "status": order.status, "total_cents": order.total_cents, "created_at": order.created_at.isoformat(), "lines": lines, "note": order.note}
        body["hash"] = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:16]
        return body

def write_invoice(order_id: int) -> Path:
    body = invoice_body(order_id)
    path = INVOICE_DIR / f"inv_{order_id}.json"
    path.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return path

def crypto_receipt(order_id: int, method: str = "local_trit") -> dict[str, Any]:
    inv = invoice_body(order_id)
    receipt = {"order_id": order_id, "method": method, "amount_cents": inv.get("total_cents", 0), "invoice_hash": inv.get("hash"), "ts": datetime.utcnow().isoformat(), "local": True, "broadcast": False}
    out = Path.home() / ".juniorhome" / "os" / "payments.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(receipt) + "\n")
    return receipt

def venmo_adapter(order_id: int, note: str = "") -> dict[str, Any]:
    inv = invoice_body(order_id)
    intent = {"order_id": order_id, "method": "venmo_intent", "amount_cents": inv.get("total_cents", 0), "note": note or f"order {order_id}", "api": "not_configured", "local": True}
    out = Path.home() / ".juniorhome" / "os" / "venmo_intents.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(intent) + "\n")
    return intent
