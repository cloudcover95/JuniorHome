#!/usr/bin/env python3
import runpy, sys
from pathlib import Path
for p in (Path("../JuniorEngrTools").resolve(), Path.home() / "JuniorCloud" / "JuniorEngrTools"):
    cli = p / "scripts" / "shop_cli.py"
    if cli.is_file():
        sys.argv = [str(cli)] + sys.argv[1:]
        runpy.run_path(str(cli), run_name="__main__")
        break
else:
    print('{"ok": false, "need": "JuniorEngrTools"}')
