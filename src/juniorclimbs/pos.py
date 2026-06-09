# path: src/juniorclimbs/pos.py

"""
JuniorClimbs POS

Production-grade Point of Sale with real ledger integration and member balance updates.
"""

from typing import Optional, Dict, Any
from .models import Member, TransactionType, PaymentMethod
from .ledger import Ledger
from .member_manager import MemberManager


class POS:
    def __init__(self, member_manager: MemberManager, ledger: Ledger):
        self.member_manager = member_manager
        self.ledger = ledger

    def sell_to_member(
        self,
        member_id: str,
        item_type: TransactionType,
        amount: float,
        payment_method: PaymentMethod,
        description: str = "",
        tax_rate: float = 0.0,
        provenance: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        member = self.member_manager.get_member(member_id)
        if not member:
            return None

        tx = self.ledger.record_transaction(
            member=member,
            tx_type=item_type,
            amount=amount,
            payment_method=payment_method,
            description=description or item_type.value,
            tax_rate=tax_rate,
            provenance=provenance,
        )

        return {
            "transaction": tx.to_dict(),
            "member_balance_after": member.current_balance,
        }

    def sell_day_pass(self, member_id: str, amount: float = 25.0, payment_method: PaymentMethod = PaymentMethod.CASH) -> Optional[Dict[str, Any]]:
        return self.sell_to_member(
            member_id=member_id,
            item_type=TransactionType.DAY_PASS,
            amount=amount,
            payment_method=payment_method,
            description="Day pass",
        )

    def sell_merch(self, member_id: str, amount: float, description: str = "Merchandise", payment_method: PaymentMethod = PaymentMethod.CASH) -> Optional[Dict[str, Any]]:
        return self.sell_to_member(
            member_id=member_id,
            item_type=TransactionType.MERCH,
            amount=amount,
            payment_method=payment_method,
            description=description,
        )

    def sell_food_drink(self, member_id: str, amount: float, description: str = "Food/Drink", payment_method: PaymentMethod = PaymentMethod.CASH) -> Optional[Dict[str, Any]]:
        return self.sell_to_member(
            member_id=member_id,
            item_type=TransactionType.FOOD_DRINK,
            amount=amount,
            payment_method=payment_method,
            description=description,
        )
