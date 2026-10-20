"""Seed general products. Local only."""
from sqlmodel import Session, SQLModel, create_engine
from pathlib import Path
from models.models import Product

DB = Path.home() / ".juniorhome" / "juniorstack.db"
engine = create_engine(f"sqlite:///{DB}")
SQLModel.metadata.create_all(engine)

seeds = [
    Product(sku="ITEM-A", name="Sample Item A", lane="general", price_cents=1000, stock=10),
    Product(sku="ITEM-B", name="Sample Item B", lane="general", price_cents=2500, stock=5),
    Product(sku="ITEM-C", name="Sample Item C", lane="general", price_cents=500, stock=20),
]

with Session(engine) as session:
    for p in seeds:
        session.add(p)
    session.commit()
print("seeded", len(seeds), "products")
