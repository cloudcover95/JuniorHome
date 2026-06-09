# path: src/juniorclimbs/member_manager.py

"""
JuniorClimbs MemberManager

Handles member lifecycle, waivers, balances, and status.
Production-grade with provenance support.
"""

from typing import Dict, Optional, List
from datetime import datetime, timedelta

from .models import Member, Waiver, MembershipStatus, CheckIn


class MemberManager:
    def __init__(self):
        self.members: Dict[str, Member] = {}
        self.waivers: Dict[str, Waiver] = {}
        self.checkins: List[CheckIn] = []

    def create_member(self, full_name: str, email: Optional[str] = None, phone: Optional[str] = None) -> Member:
        member = Member(
            full_name=full_name,
            email=email,
            phone=phone,
            status=MembershipStatus.PENDING,
        )
        self.members[member.id] = member
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

        # Mark member as active once they have a signed waiver
        if member.status == MembershipStatus.PENDING:
            member.status = MembershipStatus.ACTIVE

        return waiver

    def update_balance(self, member_id: str, amount: float, reason: str = "") -> bool:
        member = self.get_member(member_id)
        if not member:
            return False
        member.current_balance += amount
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
