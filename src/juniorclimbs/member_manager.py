# path: src/juniorclimbs/member_manager.py

"""
JuniorClimbs MemberManager

Enhanced with auto-renewal, expiry handling, and linked history tracking.
"""

from typing import Dict, Optional, List
from datetime import datetime, timedelta

from .models import Member, Waiver, MembershipStatus, CheckIn, Transaction


class MemberManager:
    def __init__(self):
        self.members: Dict[str, Member] = {}
        self.waivers: Dict[str, Waiver] = {}
        self.checkins: List[CheckIn] = []
        self.history: Dict[str, List[Dict]] = {}  # member_id -> list of events

    def create_member(self, full_name: str, email: Optional[str] = None, phone: Optional[str] = None) -> Member:
        member = Member(
            full_name=full_name,
            email=email,
            phone=phone,
            status=MembershipStatus.PENDING,
        )
        self.members[member.id] = member
        self.history[member.id] = []
        self._log_event(member.id, "member_created", {"full_name": full_name})
        return member

    def get_member(self, member_id: str) -> Optional[Member]:
        return self.members.get(member_id)

    def sign_waiver(self, member_id: str, signature_data: Optional[str] = None, ip_address: Optional[str] = None) -> Optional[Waiver]:
        member = self.get_member(member_id)
        if not member:
            return None

        waiver = Waiver(
            member_id=member_id,
            signature_data=signature_data,
            ip_address=ip_address,
        )
        self.waivers[waiver.id] = waiver

        if member.status == MembershipStatus.PENDING:
            member.status = MembershipStatus.ACTIVE

        self._log_event(member_id, "waiver_signed", {"waiver_id": waiver.id})
        return waiver

    def update_balance(self, member_id: str, amount: float, reason: str = "") -> bool:
        member = self.get_member(member_id)
        if not member:
            return False
        member.current_balance += amount
        self._log_event(member_id, "balance_updated", {"amount": amount, "reason": reason})
        return True

    def check_in(self, member_id: str, method: str = "manual") -> Optional[CheckIn]:
        member = self.get_member(member_id)
        if not member or member.status != MembershipStatus.ACTIVE:
            return None

        checkin = CheckIn(
            member_id=member_id,
            method=method,
        )
        self.checkins.append(checkin)
        self._log_event(member_id, "checkin", {"method": method})
        return checkin

    def get_active_members(self) -> List[Member]:
        return [m for m in self.members.values() if m.status == MembershipStatus.ACTIVE]

    def get_members_with_expiring_membership(self, days: int = 7) -> List[Member]:
        now = datetime.utcnow()
        threshold = now + timedelta(days=days)
        return [
            m for m in self.members.values()
            if m.membership_expires and m.membership_expires <= threshold
        ]

    def auto_renew_membership(self, member_id: str, months: int = 1) -> bool:
        member = self.get_member(member_id)
        if not member:
            return False

        if member.membership_expires:
            member.membership_expires += timedelta(days=30 * months)
        else:
            member.membership_expires = datetime.utcnow() + timedelta(days=30 * months)

        member.status = MembershipStatus.ACTIVE
        self._log_event(member_id, "auto_renewed", {"months": months})
        return True

    def _log_event(self, member_id: str, event_type: str, data: Dict):
        if member_id not in self.history:
            self.history[member_id] = []
        self.history[member_id].append({
            "timestamp": datetime.utcnow().isoformat(),
            "event": event_type,
            "data": data
        })

    def get_member_history(self, member_id: str) -> List[Dict]:
        return self.history.get(member_id, [])
