#!/usr/bin/env python3
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web3node"))
from mobile_host import serve, ticket

if __name__ == "__main__":
    if "--serve" in sys.argv:
        serve()
    else:
        print(json.dumps(ticket(), indent=2))
