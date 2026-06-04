# path: src/juniorclimbs/agent/integration.py

class JuniorClimbsAgentInterface:
    def __init__(self, pos, events, safety, ledger, maintenance, data_store):
        self.pos = pos
        self.events = events
        self.safety = safety
        self.ledger = ledger
        self.maintenance = maintenance
        self.data_store = data_store

    def get_member_status(self, member_id: str) -> dict:
        return self.data_store.get_member(member_id)

    def check_safety_at_location(self, location: dict) -> list:
        return self.safety.check_point_safety(location)

    def book_class_for_member(self, member_id: str, event_id: str):
        return self.events.book_class(event_id, member_id)

    def apply_discount(self, member_id: str, percent: float, reason: str):
        return self.ledger.inject_discount(member_id, percent, reason)

    def log_maintenance(self, equipment_id: str, action: str, performed_by: str, notes: str):
        return self.maintenance.add_log(equipment_id, action, performed_by, notes)

    def get_maintenance_overdue(self):
        return self.maintenance.get_overdue()

    def get_daily_schedule(self, date):
        return self.events.get_daily_schedule(date)
