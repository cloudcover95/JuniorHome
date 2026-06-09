# path: src/juniorclimbs/cli.py

"""
JuniorClimbs CLI

Simple, fast command-line interface for Linux beta testing and gym staff use.
Lightning-fast by design. BitNet 1.58/3.0 ready for future inference paths.
"""

import argparse
import sys
from datetime import datetime

from . import (
    MemberManager,
    Ledger,
    POS,
    SafetyManager,
    Reporter,
    WaiverManager,
    TransactionType,
    PaymentMethod,
)


def main():
    parser = argparse.ArgumentParser(description="JuniorClimbs - Production Gym Management (Linux Beta)")
    subparsers = parser.add_subparsers(dest="command")

    # Member commands
    member_parser = subparsers.add_parser("member", help="Member operations")
    member_sub = member_parser.add_subparsers(dest="action")
    member_sub.add_parser("list", help="List active members")
    member_sub.add_parser("create", help="Create new member").add_argument("name", help="Full name")

    # Check-in
    checkin_parser = subparsers.add_parser("checkin", help="Check in a member")
    checkin_parser.add_argument("member_id", help="Member ID or name search")

    # Waiver
    waiver_parser = subparsers.add_parser("waiver", help="Waiver operations")
    waiver_parser.add_argument("--generate-qr", action="store_true", help="Generate QR payload for new waiver")

    # POS
    pos_parser = subparsers.add_parser("pos", help="Point of Sale")
    pos_parser.add_argument("member_id")
    pos_parser.add_argument("item", choices=["day_pass", "merch", "food"])
    pos_parser.add_argument("amount", type=float)

    # Safety
    safety_parser = subparsers.add_parser("safety", help="Wall/area status")
    safety_parser.add_argument("area_id")
    safety_parser.add_argument("status", choices=["open", "restricted", "closed"])

    # Report
    report_parser = subparsers.add_parser("report", help="Basic reports")
    report_parser.add_argument("type", choices=["daily", "expiring"])

    args = parser.parse_args()

    # Initialize core systems
    member_manager = MemberManager()
    ledger = Ledger()
    pos = POS(member_manager, ledger)
    safety = SafetyManager()
    reporter = Reporter(member_manager, ledger)
    waiver_mgr = WaiverManager(member_manager)

    if args.command == "member":
        if args.action == "list":
            for m in member_manager.get_active_members():
                print(f"{m.id} | {m.full_name} | Balance: {m.current_balance}")
        elif args.action == "create":
            member = member_manager.create_member(args.name)
            print(f"Created member: {member.id} - {member.full_name}")

    elif args.command == "checkin":
        # Simple seamless check-in (waiver check would be added in real flow)
        print(f"Checked in member: {args.member_id} (simulated seamless)")

    elif args.command == "waiver":
        if args.generate_qr:
            payload = waiver_mgr.generate_qr_payload()
            print("QR Payload for mobile waiver:")
            print(payload)

    elif args.command == "pos":
        item_map = {
            "day_pass": TransactionType.DAY_PASS,
            "merch": TransactionType.MERCH,
            "food": TransactionType.FOOD_DRINK,
        }
        result = pos.sell_to_member(
            args.member_id,
            item_map[args.item],
            args.amount,
            PaymentMethod.CASH,
        )
        print(result)

    elif args.command == "safety":
        area = safety.update_area_status(args.area_id, args.status)
        print(f"Updated area {args.area_id} -> {args.status}")

    elif args.command == "report":
        if args.type == "daily":
            print(reporter.daily_revenue_report())
        elif args.type == "expiring":
            print(reporter.renewals_due_report())

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
