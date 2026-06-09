# path: src/juniorclimbs/ledger.py

"""
JuniorClimbs Ledger

Production-grade transaction ledger with tax-ready output and balance updates.
Designed for real business operations.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime

from .models import Transaction, TransactionType, PaymentMethod, Member


class Ledger:
    def __init__(self):
        self.transactions: List[Transaction] = []
        self.daily_totals: Dict[str, float] = {}  # date_str -> total

    def record_transaction(
        self,
        member: Member,
        tx_type: TransactionType,
        amount: float,
        payment_method: PaymentMethod,
        description: str = "",
        tax_rate: float = 0.0,
        provenance: Optional[Dict[str, Any]] = None
    ) -> Transaction:
        tax_amount = round(amount * tax_rate, 2)
        total_amount = amount + tax_amount

        # Update member balance
        if payment_method == PaymentMethod.ACCOUNT_BALANCE:
            member.current_balance -= total_amount
        else:
            # For cash/card/etc, we still track but don't deduct from internal balance
            pass

        tx = Transaction(
            member_id=member.id,
            type=tx_type,
            amount=total_amount,
            payment_method=payment_method,
            description=description,
            tax_amount=tax_amount,
            balance_after=member.current_balance,
            provenance=provenance or {},
        )

        self.transactions.append(tx)

        # Update daily totals
        date_key = tx.timestamp.date().isoformat()
        self.daily_totals[date_key] = self.daily_totals.get(date_key, 0.0) + total_amount

        return tx

    def get_member_transactions(self, member_id: str) -> List[Transaction]:
        return [tx for tx in self.transactions if tx.member_id == member_id]

    def get_daily_revenue(self, date_str: Optional[str] = None) -> float:
        if date_str:
            return self.daily_totals.get(date_str, 0.0)
        return sum(self.daily_totals.values())

    def export_tax_ready(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> List[Dict[str, Any]]:
        """Export transactions in a tax/accounting friendly format."""
        filtered = self.transactions
        if start_date or end_date:
            filtered = [
                tx for tx in self.transactions
                if (not start_date or tx.timestamp.date().isoformat() >= start_date)
                and (not end_date or tx.timestamp.date().isoformat() <= end_date)
            ]
        return [tx.to_dict() for tx in filtered]

    def get_member_balance(self, member: Member) -> float:
        return member.current_balance
