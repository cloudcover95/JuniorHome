"""Suite status includes the deny-path check."""
import sys
from gates_test import check
def suite():
    return {"gates": check()["ok"], "bind": "127.0.0.1", "model_pull": False}
if __name__ == "__main__":
    import json
    row = suite()
    print(json.dumps(row, indent=2))
    sys.exit(0 if row["gates"] else 1)
