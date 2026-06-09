# path: src/juniorclimbs/employee.py

"""
JuniorClimbs Employee & Scheduling

Lightning-fast, lean employee management aligned with BitNet-native efficiency.
8-hour shifts = 30min lunch. Other shifts = 15min breaks.
Auto notifications (email/slack/telegram configurable).
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
from enum import Enum

import uuid


class ShiftType(str, Enum):
    EIGHT_HOUR = "8_hour"
    FOUR_HOUR = "4_hour"
    OTHER = "other"


@dataclass
class Employee:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    full_name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    role: str = "staff"  # staff, manager, admin
    created_at: datetime = field(default_factory=datetime.utcnow)
    active: bool = True
    notification_preference: str = "email"  # email, slack, telegram


@dataclass
class Shift:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    employee_id: str
    start_time: datetime
    end_time: datetime
    shift_type: ShiftType
    break_minutes: int = 0
    notes: str = ""

    def get_break_info(self) -> str:
        if self.shift_type == ShiftType.EIGHT_HOUR:
            return "30 min lunch break"
        else:
            return "15 min break"


class ScheduleManager:
    def __init__(self):
        self.employees: Dict[str, Employee] = {}
        self.shifts: List[Shift] = []

    def add_employee(self, full_name: str, email: Optional[str] = None, role: str = "staff") -> Employee:
        emp = Employee(full_name=full_name, email=email, role=role)
        self.employees[emp.id] = emp
        return emp

    def create_shift(self, employee_id: str, start_time: datetime, end_time: datetime, shift_type: ShiftType = ShiftType.EIGHT_HOUR) -> Optional[Shift]:
        if employee_id not in self.employees:
            return None

        break_min = 30 if shift_type == ShiftType.EIGHT_HOUR else 15

        shift = Shift(
            employee_id=employee_id,
            start_time=start_time,
            end_time=end_time,
            shift_type=shift_type,
            break_minutes=break_min,
        )
        self.shifts.append(shift)
        return shift

    def get_employee_shifts(self, employee_id: str, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> List[Shift]:
        shifts = [s for s in self.shifts if s.employee_id == employee_id]
        if start_date:
            shifts = [s for s in shifts if s.start_time >= start_date]
        if end_date:
            shifts = [s for s in shifts if s.end_time <= end_date]
        return shifts

    def generate_schedule_legend(self) -> str:
        """Small legend for printed/exported schedules (minimal space)."""
        return "Legend: 8hr shifts = 30min lunch | Other shifts = 15min break"

    def get_upcoming_shifts(self, days: int = 14) -> List[Shift]:
        now = datetime.utcnow()
        future = now + timedelta(days=days)
        return [s for s in self.shifts if now <= s.start_time <= future]
