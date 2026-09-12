"""junior_arm_iot — TritARM + IoT host pack for JuniorHome.

Stdlib. BitNet scores CPU intent only. Not a Nintendo product.
"""
from . import host

Machine = host.Machine
PROFILES = host.PROFILES
get_profile = host.get_profile
assemble = host.assemble
score_intent = host.score_intent
retarget_pad = host.retarget_pad

__all__ = [
    "Machine",
    "PROFILES",
    "get_profile",
    "assemble",
    "score_intent",
    "retarget_pad",
    "host",
]
classify_bytes = host.classify_bytes
sis_commit = host.sis_commit
apply_crispy = host.apply_crispy
xr_pose = host.xr_pose
engine_route = host.engine_route

__version__ = "0.2.0"
