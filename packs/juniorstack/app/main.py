"""JuniorStack FastAPI app. Local ops + live storefront."""
from __future__ import annotations
from pathlib import Path
from fastapi import FastAPI, Request, Form, Depends
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlmodel import Session, SQLModel, create_engine, select
from models.models import Product, Order, OrderItem
from models.payments import write_invoice, crypto_receipt, venmo_adapter
from models.comms import advance, message
from models.agent_fulfill import check_addresses, agent_fulfill
from models.sale_stack import on_sale
from scripts.brain_quant import digest

DB = Path.home() / ".juniorhome" / "juniorstack.db"
DB.parent.mkdir(parents=True, exist_ok=True)
engine = create_engine(f"sqlite:///{DB}", echo=False)

def get_session():
    with Session(engine) as session:
        yield session

app = FastAPI(title="JuniorStack", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="app/templates")

@app.on_event("startup")
def startup():
    SQLModel.metadata.create_all(engine)

@app.get("/", response_class=HTMLResponse)
def catalog(request: Request, session: Session = Depends(get_session)):
    products = session.exec(select(Product).where(Product.active == True)).all()
    return templates.TemplateResponse("catalog.html", {"request": request, "products": products})

@app.get("/admin", response_class=HTMLResponse)
def admin(request: Request, session: Session = Depends(get_session)):
    products = session.exec(select(Product)).all()
    orders = session.exec(select(Order).order_by(Order.created_at.desc())).all()
    return templates.TemplateResponse("admin.html", {"request": request, "products": products, "orders": orders})

@app.post("/admin/product")
def add_product(sku: str = Form(...), name: str = Form(...), lane: str = Form(...), price_cents: int = Form(...), stock: int = Form(0), session: Session = Depends(get_session)):
    p = Product(sku=sku, name=name, lane=lane, price_cents=price_cents, stock=stock)
    session.add(p)
    session.commit()
    return RedirectResponse("/admin", status_code=303)

@app.post("/admin/advance")
def admin_advance(order_id: int = Form(...), status: str = Form(...)):
    return advance(order_id, status)

@app.post("/admin/invoice")
def admin_invoice(order_id: int = Form(...)):
    path = write_invoice(order_id)
    return {"invoice": str(path)}

@app.post("/admin/crypto")
def admin_crypto(order_id: int = Form(...)):
    return crypto_receipt(order_id)

@app.post("/admin/venmo")
def admin_venmo(order_id: int = Form(...), note: str = Form("")):
    return venmo_adapter(order_id, note)

@app.post("/admin/message")
def admin_message(order_id: int = Form(...), text: str = Form(...)):
    path = message(order_id, text)
    return {"message": str(path)}

@app.get("/admin/brain")
def admin_brain():
    return digest()

@app.get("/agent/crypto")
def agent_crypto():
    return check_addresses()

@app.post("/agent/fulfill")
def agent_fulfill_ep(order_id: int = Form(...), method: str = Form("local_trit")):
    return agent_fulfill(order_id, method)

@app.post("/order")
def place_order(product_id: int = Form(...), qty: int = Form(1), session: Session = Depends(get_session)):
    product = session.get(Product, product_id)
    if not product or product.stock < qty:
        return RedirectResponse("/", status_code=303)
    order = Order(status="received", total_cents=product.price_cents * qty)
    session.add(order)
    session.commit()
    session.refresh(order)
    item = OrderItem(order_id=order.id, product_id=product.id, qty=qty, price_cents=product.price_cents)
    product.stock -= qty
    session.add(item)
    session.commit()
    on_sale(order.id, product.sku, product.name, qty)
    return RedirectResponse("/", status_code=303)
