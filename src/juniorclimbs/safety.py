# path: src/juniorclimbs/safety.py

"""
JuniorClimbs SafetyManager

More robust wall/area status with staff override and area-tied maintenance.
"""

from typing import Dict, List, Optional
from datetime import datetime

from .models import WallArea, MaintenanceLog


class SafetyManager:
    def __init__(self):
        self.areas: Dict[str, WallArea] = {}
        self.maintenance_logs: List[MaintenanceLog] = []

    def create_area(self, name: str, area_type: str) -> WallArea:
        area = WallArea(name=name, area_type=area_type)
        self.areas[area.id] = area
        return area

    def update_area_status(self, area_id: str, status: str, restrictions: Optional[List[str]] = None, updated_by: Optional[str] = None, override: bool = False) -> Optional[WallArea]:
        area = self.areas.get(area_id)
        if not area:
            return None

        # Staff override allowed
        if override or status in ["open", "restricted", "closed"]:
            area.status = status
            if restrictions is not None:
                area.restrictions = restrictions
            area.last_updated = datetime.utcnow()
            area.updated_by = updated_by
        return area

    def get_area(self, area_id: str) -> Optional[WallArea]:
        return self.areas.get(area_id)

    def get_all_areas(self) -> List[WallArea]:
        return list(self.areas.values())

    def log_maintenance(self, area_id: str, description: str, performed_by: Optional[str] = None) -> MaintenanceLog:
        log = MaintenanceLog(
            area_id=area_id,
            description=description,
            performed_by=performed_by,
        )
        self.maintenance_logs.append(log)
        return log

    def get_maintenance_logs(self, area_id: Optional[str] = None) -> List[MaintenanceLog]:
        if area_id:
            return [log for log in self.maintenance_logs if log.area_id == area_id]
        return self.maintenance_logs
