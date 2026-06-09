# path: src/juniorclimbs/web.py

"""
JuniorClimbs Web Interface (Production Beta)

Simple, clean, point-and-click web UI for regular gym staff.
No terminal required for daily operations.
Lightning fast + BitNet-ready architecture.
"""

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from typing import Optional

import os

from .member_manager import MemberManager
from .ledger import Ledger
from .pos import POS
from .safety import SafetyManager
from .waiver import WaiverManager
from .employee import ScheduleManager
from .events import EventManager

from .models import TransactionType, PaymentMethod

app = FastAPI(title="JuniorClimbs - Gym Operations")

templates = Jinja2Templates(directory="src/juniorclimbs/templates")

# Core systems (in production these would be properly initialized / persisted)
mm = MemberManager()
ledger = Ledger()
pos_system = POS(mm, ledger)
safety = SafetyManager()
wm = WaiverManager(mm)
sm = ScheduleManager()
em = EventManager()

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    active_members = len(mm.get_active_members())
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "active_members": active_members,
        "title": "JuniorClimbs Dashboard"
    })

@app.get("/checkin", response_class=HTMLResponse)
async def checkin_page(request: Request):
    return templates.TemplateResponse("checkin.html", {
        "request": request,
        "title": "Member Check-in"
    })

@app.post("/checkin")
async def do_checkin(member_id: str = Form(...)):
    member = mm.get_member(member_id)
    if member:
        mm.check_in(member_id)
        return RedirectResponse("/checkin?success=1", status_code=303)
    return RedirectResponse("/checkin?error=not_found", status_code=303)

@app.get("/pos", response_class=HTMLResponse)
async def pos_page(request: Request):
    return templates.TemplateResponse("pos.html", {
        "request": request,
        "title": "Point of Sale"
    })

@app.post("/pos")
async def do_pos(
    member_id: str = Form(...),
    item_type: str = Form(...),
    amount: float = Form(0.0)
):
    item_map = {
        "day_pass": TransactionType.DAY_PASS,
        "merch": TransactionType.MERCH,
        "food": TransactionType.FOOD_DRINK,
    }
    tx_type = item_map.get(item_type, TransactionType.OTHER)
    pos_system.sell_to_member(member_id, tx_type, amount, PaymentMethod.CASH)
    return RedirectResponse("/pos?success=1", status_code=303)

@app.get("/safety", response_class=HTMLResponse)
async def safety_page(request: Request):
    areas = safety.get_all_areas()
    return templates.TemplateResponse("safety.html", {
        "request": request,
        "areas": areas,
        "title": "Safety Zones"
    })

@app.post("/safety/update")
async def update_safety(area_id: str = Form(...), status: str = Form(...)):
    safety.update_area_status(area_id, status, override=True)
    return RedirectResponse("/safety", status_code=303)

@app.get("/members", response_class=HTMLResponse)
async def members_page(request: Request):
    members = mm.get_active_members()
    return templates.TemplateResponse("members.html", {
        "request": request,
        "members": members,
        "title": "Members"
    })

# Simple health check for Docker / monitoring
@app.get("/health")
async def health():
    return {"status": "ok"}
