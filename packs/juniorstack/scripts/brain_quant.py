"""Brain quant hook for JuniorStack orders. Local only."""
from __future__ import annotations
import json
from pathlib import Path
from sqlmodel import Session, create_engine, select
from models.models import Order, Product

DB = Path.home() / ".juniorhome" / "juniorstack.db"

def digest() -> dict:
    if not DB.exists():
        return {"ok": False}
    engine = create_engine(f"sqlite:///{DB}")
    with Session(engine) as session:
        orders = session.exec(select(Order)).all()
        products = session.exec(select(Product)).all()
        by_lane = {}
        for p in products:
            by_lane.setdefault(p.lane, 0)
            by_lane[p.lane] += p.stock
        body = {"orders": len(orders), "open": sum(1 for o in orders if o.status != "done"), "stock_by_lane": by_lane, "local": True}
    out = Path.home() / ".juniorhome" / "os" / "juniorstack_brain.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return body

if __name__ == "__main__":
    print(json.dumps(digest(), indent=2))
