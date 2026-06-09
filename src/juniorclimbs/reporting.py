# path: src/juniorclimbs/reporting.py

"""
JuniorClimbs Reporting

Basic but production-useful reports for gym operations.
"""

from typing import List, Dict, Any
from datetime import datetime, timedelta

from .member_manager import MemberManager
from .ledger import Ledger


class Reporter:
    def __init__(self, member_manager: MemberManager, ledger: Ledger):
        self.member_manager = member_manager
        self.ledger = ledger

    def daily_revenue_report(self, date_str: Optional[str] = None) -> Dict[str, Any]:
        total = self.ledger.get_daily_revenue(date_str)
        return {
            "date": date_str or "all_time",
            "total_revenue": total,
            "transaction_count": len(self.ledger.transactions),
        }

    def member_activity_summary(self) -> Dict[str, Any]:
        active = len(self.member_manager.get_active_members())
        total_members = len(self.member_manager.members)
        expiring_soon = len(self.member_manager.get_members_with_expiring_membership(7))

        return {
            "total_members": total_members,
            "active_members": active,
            "expiring_in_7_days": expiring_soon,
        }

    def renewals_due_report(self, days: int = 7) -> List[Dict[str, Any]]:
        members = self.member_manager.get_members_with_expiring_membership(days)
        return [
            {
                "id": m.id,
                "name": m.full_name,
                "email": m.email,
                "expires": m.membership_expires.isoformat() if m.membership_expires else None,
            }
            for m in members
        ]
