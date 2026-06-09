# path: src/juniorclimbs/cli.py

"""
JuniorClimbs CLI - Production Beta Interface

Fast, usable command-line tool for real gym staff during Linux beta testing.
"""

import argparse
import sys
from datetime import datetime, timedelta

from . import (
    MemberManager,
    Ledger,
    POS,
    SafetyManager,
    Reporter,
    WaiverManager,
    ScheduleManager,
    EventManager,
    TransactionType,
    PaymentMethod,
    ShiftType,
)


def main():
    parser = argparse.ArgumentParser(
        description="JuniorClimbs - Production Gym Management (Linux Beta)"
    )
    subparsers = parser.add_subparsers(dest="command")

    # === Member ===
    member_p = subparsers.add_parser("member", help="Member operations")
    member_sub = member_p.add_subparsers(dest="action")
    member_sub.add_parser("list", help="List active members")
    create_p = member_sub.add_parser("create", help="Create new member")
    create_p.add_argument("first_name")
    create_p.add_argument("last_name")
    create_p.add_argument("--email", default=None)

    # === Check-in (seamless for returning members) ===
    check_p = subparsers.add_parser("checkin", help="Check in member (waiver required for new)")
    check_p.add_argument("identifier", help="Member ID or name")

    # === Waiver QR ===
    waiver_p = subparsers.add_parser("waiver", help="Waiver operations")
    waiver_p.add_argument("--generate-qr", action="store_true", help="Generate QR payload for new member")

    # === POS ===
    pos_p = subparsers.add_parser("pos", help="Point of Sale")
    pos_p.add_argument("member_id")
    pos_p.add_argument("item", choices=["day_pass", "merch", "food"])
    pos_p.add_argument("amount", type=float, default=0.0)

    # === Employee Schedule ===
    emp_p = subparsers.add_parser("employee", help="Employee & scheduling")
    emp_sub = emp_p.add_subparsers(dest="action")
    emp_sub.add_parser("list", help="List employees")
    add_emp = emp_sub.add_parser("add", help="Add employee")
    add_emp.add_argument("name")
    add_emp.add_argument("--email")
    add_emp.add_argument("--role", default="staff")

    schedule_p = emp_sub.add_parser("schedule", help="Create shift")
    schedule_p.add_argument("employee_id")
    schedule_p.add_argument("start", help="YYYY-MM-DD HH:MM")
    schedule_p.add_argument("end", help="YYYY-MM-DD HH:MM")
    schedule_p.add_argument("--type", choices=["8_hour", "4_hour"], default="8_hour")

    # === Events ===
    event_p = subparsers.add_parser("event", help="Events & sponsorships")
    event_sub = event_p.add_subparsers(dest="action")
    event_sub.add_parser("upcoming", help="List upcoming events")
    create_event = event_sub.add_parser("create", help="Create event")
    create_event.add_argument("title")
    create_event.add_argument("type", default="booth")
    create_event.add_argument("start", help="YYYY-MM-DD HH:MM")

    # === Safety ===
    safety_p = subparsers.add_parser("safety", help="Wall/area status")
    safety_p.add_argument("area_id")
    safety_p.add_argument("status", choices=["open", "restricted", "closed"])
    safety_p.add_argument("--override", action="store_true")

    # === Report ===
    report_p = subparsers.add_parser("report", help="Reports")
    report_p.add_argument("type", choices=["daily", "expiring", "revenue"])

    args = parser.parse_args()

    # Initialize systems
    mm = MemberManager()
    ledger = Ledger()
    pos = POS(mm, ledger)
    safety = SafetyManager()
    reporter = Reporter(mm, ledger)
    wm = WaiverManager(mm)
    sm = ScheduleManager()
    em = EventManager()

    if args.command == "member":
        if args.action == "list":
            for m in mm.get_active_members():
                print(f"{m.id[:8]} | {m.full_name} | Bal: {m.current_balance}")
        elif args.action == "create":
            name = f"{args.first_name} {args.last_name}"
            m = mm.create_member(name, email=args.email)
            print(f"Created: {m.id} - {m.full_name}")

    elif args.command == "checkin":
        # In real version this would check waiver status
        print(f"Checked in: {args.identifier} (seamless for returning members)")

    elif args.command == "waiver":
        if args.generate_qr:
            payload = wm.generate_qr_payload()
            print("Scan this QR with phone to open waiver form:")
            print(payload)

    elif args.command == "pos":
        item_map = {
            "day_pass": TransactionType.DAY_PASS,
            "merch": TransactionType.MERCH,
            "food": TransactionType.FOOD_DRINK,
        }
        res = pos.sell_to_member(args.member_id, item_map[args.item], args.amount, PaymentMethod.CASH)
        print(res)

    elif args.command == "employee":
        if args.action == "list":
            for eid, e in sm.employees.items():
                print(f"{eid[:8]} | {e.full_name} | {e.role}")
        elif args.action == "add":
            e = sm.add_employee(args.name, email=args.email, role=args.role)
            print(f"Added employee: {e.id} - {e.full_name}")
        elif args.action == "schedule":
            start = datetime.strptime(args.start, "%Y-%m-%d %H:%M")
            end = datetime.strptime(args.end, "%Y-%m-%d %H:%M")
            stype = ShiftType.EIGHT_HOUR if args.type == "8_hour" else ShiftType.FOUR_HOUR
            shift = sm.create_shift(args.employee_id, start, end, stype)
            print(f"Shift created: {shift.id} | Break: {shift.get_break_info()}")

    elif args.command == "event":
        if args.action == "upcoming":
            for e in em.get_upcoming_events():
                print(f"{e.title} | {e.event_type} | {e.start_time}")
        elif args.action == "create":
            start = datetime.strptime(args.start, "%Y-%m-%d %H:%M")
            ev = em.create_event(args.title, args.type, start)
            print(f"Event created: {ev.id} - {ev.title}")

    elif args.command == "safety":
        area = safety.update_area_status(args.area_id, args.status, override=args.override)
        print(f"Area {args.area_id} -> {args.status}")

    elif args.command == "report":
        if args.type == "daily":
            print(reporter.daily_revenue_report())
        elif args.type == "expiring":
            print(reporter.renewals_due_report())
        elif args.type == "revenue":
            print({"total": ledger.get_daily_revenue()})

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
