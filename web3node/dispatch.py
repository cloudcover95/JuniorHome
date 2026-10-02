"""Dispatch one job. Absent class denies."""
from route import route
def dispatch(job="ticket"):
    present = {"pi": False, "gpu": False, "spark": False, "mac": False}
    row = route(job, present)
    row["bind"] = "127.0.0.1"
    return row
