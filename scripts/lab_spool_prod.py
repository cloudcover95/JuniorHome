#!/usr/bin/env python3
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ports.lab_catalog import CATALOG
from ports.lab_spool import list_spool, queue

cmd = sys.argv[1] if len(sys.argv) > 1 else "list"
if cmd == "queue" and len(sys.argv) >= 4:
    print(json.dumps(queue(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else "(empty)\n"), indent=2))
elif cmd == "catalog":
    print(json.dumps(CATALOG, indent=2))
else:
    print(json.dumps({"spool": list_spool(), "fire": False}, indent=2))
