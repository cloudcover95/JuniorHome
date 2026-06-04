# path: src/juniorhome/junioragi/task_types.py
#!/usr/bin/env python3
"""
JuniorAGI Task Types

Concrete, end-user friendly task definitions that map to the
Tri-State Execution Paradigm + kernel injection.
"""

from enum import Enum


class JuniorAGITaskType(str, Enum):
    # User Black Box
    USER_ANALYZE = "user_analyze"
    USER_FOLD_MANIFOLD = "user_fold_manifold"

    # Swarm Black Box
    SWARM_DEBATE = "swarm_debate"
    SWARM_LEARN = "swarm_learn"

    # Industry Fallback
    INDUSTRY_VERIFY = "industry_verify"

    # Kernel / System
    KERNEL_INJECT = "kernel_inject"
    DIAGNOSTIC = "diagnostic"
    BUILD_TEST = "build_test"

    # Convenience
    AUTO = "auto"


TASK_TO_MODE = {
    JuniorAGITaskType.USER_ANALYZE: "user",
    JuniorAGITaskType.USER_FOLD_MANIFOLD: "user",
    JuniorAGITaskType.SWARM_DEBATE: "swarm",
    JuniorAGITaskType.SWARM_LEARN: "swarm",
    JuniorAGITaskType.INDUSTRY_VERIFY: "industry",
    JuniorAGITaskType.KERNEL_INJECT: "user",  # defaults to user box then injects
    JuniorAGITaskType.DIAGNOSTIC: "auto",
    JuniorAGITaskType.BUILD_TEST: "auto",
    JuniorAGITaskType.AUTO: "auto",
}
