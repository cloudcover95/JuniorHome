# path: src/juniorclimbs/events.py

"""
JuniorClimbs Events & Sponsorships

Lightweight event, booth, partnership, and incentive tracking.
Aligned with efficient core architecture.
"""

from typing import Dict, List, Optional
from datetime import datetime

import uuid


@dataclass
class Event:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    event_type: str  # booth, sponsorship, company_meeting, partner_visit, etc.
    start_time: datetime
    end_time: Optional[datetime] = None
    location: str = ""
    description: str = ""
    partner_brand: Optional[str] = None
    incentive: Optional[str] = None  # e.g. "10% discount for members"
    created_at: datetime = field(default_factory=datetime.utcnow)


class EventManager:
    def __init__(self):
        self.events: Dict[str, Event] = {}

    def create_event(
        self,
        title: str,
        event_type: str,
        start_time: datetime,
        end_time: Optional[datetime] = None,
        location: str = "",
        partner_brand: Optional[str] = None,
        incentive: Optional[str] = None
    ) -> Event:
        event = Event(
            title=title,
            event_type=event_type,
            start_time=start_time,
            end_time=end_time,
            location=location,
            partner_brand=partner_brand,
            incentive=incentive,
        )
        self.events[event.id] = event
        return event

    def get_upcoming_events(self, days: int = 30) -> List[Event]:
        now = datetime.utcnow()
        future = now + timedelta(days=days)
        return [e for e in self.events.values() if e.start_time >= now and e.start_time <= future]

    def get_events_by_type(self, event_type: str) -> List[Event]:
        return [e for e in self.events.values() if e.event_type == event_type]
