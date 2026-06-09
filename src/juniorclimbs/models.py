# path: src/juniorclimbs/models.py

"""
JuniorClimbs Core Data Models

Production-grade, type-safe models for real gym operations.
Strong provenance support for auditability and future Obsidian export.
"""

from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Optional, List, Dict, Any
from enum import Enum

import uuid


class MembershipStatus(str, Enum):
    ACTIVE = "active"
    EXPIRED = "expired"
    SUSPENDED = "suspended"
    PENDING = "pending"


class TransactionType(str, Enum):
    MEMBERSHIP = "membership"
    DAY_PASS = "day_pass"
    MERCH = "merch"
    FOOD_DRINK = "food_drink"
    OTHER = "other"


class PaymentMethod(str, Enum):
    CARD = "card"
    CASH = "cash"
    ACCOUNT_BALANCE = "account_balance"
    OTHER = "other"


@dataclass
class Member:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    full_name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    date_of_birth: Optional[date] = None
    emergency_contact: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    status: MembershipStatus = MembershipStatus.PENDING
    current_balance: float = 0.0
    membership_tier: Optional[str] = None
    membership_expires: Optional[datetime] = None
    notes: str = ""
    provenance: Dict[str, Any] = field(default_factory=dict)  # For BitNet/GraphMemory links

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "full_name": self.full_name,
            "email": self.email,
            "phone": self.phone,
            "status": self.status.value,
            "current_balance": self.current_balance,
            "membership_tier": self.membership_tier,
            "membership_expires": self.membership_expires.isoformat() if self.membership_expires else None,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class MembershipTier:
    name: str
    price_monthly: float
    price_annual: Optional[float] = None
    benefits: List[str] = field(default_factory=list)
    auto_renew: bool = True


@dataclass
class Transaction:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    member_id: str
    type: TransactionType
    amount: float
    payment_method: PaymentMethod
    timestamp: datetime = field(default_factory=datetime.utcnow)
    description: str = ""
    tax_amount: float = 0.0
    balance_after: float = 0.0
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "member_id": self.member_id,
            "type": self.type.value,
            "amount": self.amount,
            "payment_method": self.payment_method.value,
            "timestamp": self.timestamp.isoformat(),
            "description": self.description,
            "tax_amount": self.tax_amount,
            "balance_after": self.balance_after,
        }


@dataclass
class Waiver:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    member_id: str
    signed_at: datetime = field(default_factory=datetime.utcnow)
    ip_address: Optional[str] = None
    signature_data: Optional[str] = None  # Can store hash or base64 of signature
    version: str = "v1.0"
    content_hash: Optional[str] = None
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "member_id": self.member_id,
            "signed_at": self.signed_at.isoformat(),
            "version": self.version,
        }


@dataclass
class CheckIn:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    member_id: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    method: str = "manual"  # qr, manual, keyfob, etc.
    location: str = "main"
    provenance: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WallArea:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    area_type: str  # slab, overhang, trad, top_out, etc.
    status: str = "open"  # open, restricted, closed, maintenance
    restrictions: List[str] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    updated_by: Optional[str] = None


@dataclass
class MaintenanceLog:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    area_id: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    description: str
    performed_by: Optional[str] = None
    resolved: bool = False
