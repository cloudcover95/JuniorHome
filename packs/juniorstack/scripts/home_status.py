"""JuniorHome status hook for JuniorStack. Local only."""
from __future__ import annotations
import json
from pathlib import Path
from sqlmodel import Session, create_engine, select
from models.models import Product, Order

DB = Path.home() / ".juniorhome" / "juniorstack.db"
if not DB.exists():
    print("no db")
    raise SystemExit(0)

engine = create_engine(f"sqlite:///{DB}")
with Session(engine) as session:
    low = session.exec(select(Product).where(Product.stock < 3, Product.active == True)).all()
    open_orders = session.exec(select(Order).where(Order.status != "done")).all()

body = {"low_stock": [{"sku": p.sku, "stock": p.stock} for p in low], "open_orders": len(open_orders), "local": True}
out = Path.home() / ".juniorhome" / "os" / "juniorstack.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(body, indent=2), encoding="utf-8")
print(json.dumps(body, indent=2))
