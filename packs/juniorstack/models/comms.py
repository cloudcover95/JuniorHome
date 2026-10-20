"""Order communication and status automation. Local only."""
from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
from sqlmodel import Session, create_engine
from models.models import Order
from models.payments import write_invoice
from models.trit import note_line

DB = Path.home() / ".juniorhome" / "juniorstack.db"
MSG_DIR = Path.home() / ".juniorhome" / "juniorstack" / "messages"
MSG_DIR.mkdir(parents=True, exist_ok=True)

STATUSES = ["received", "making", "shipped", "done"]

def _engine():
    return create_engine(f"sqlite:///{DB}")

def advance(order_id: int, to: str) -> dict:
    if to not in STATUSES:
        return {"ok": False, "reason": "bad status"}
    engine = _engine()
    with Session(engine) as session:
        order = session.get(Order, order_id)
        if not order:
            return {"ok": False, "reason": "no order"}
        order.status = to
        session.add(order)
        session.commit()
        write_invoice(order_id)
        line = note_line(to, order_id)
        msg = {"order_id": order_id, "status": to, "ts": datetime.utcnow().isoformat(), "text": f"Order {order_id} is now {to}.", "pack": line["pack"]}
        path = MSG_DIR / f"order_{order_id}.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(msg) + "\n")
        return {"ok": True, "order_id": order_id, "status": to, "pack": line["pack"]}

def message(order_id: int, text: str) -> Path:
    msg = {"order_id": order_id, "ts": datetime.utcnow().isoformat(), "text": text}
    path = MSG_DIR / f"order_{order_id}.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(msg) + "\n")
    return path
