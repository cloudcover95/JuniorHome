#!/usr/bin/env python3
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ports.pointers import list_pointers
print(json.dumps(list_pointers(), indent=2))
