"""Write a JuniorDeck Ardour session drop. Does not vendor Ardour. Does not launch it."""
from __future__ import annotations
import json, shutil
from pathlib import Path
OUT = Path.home() / ".juniorhome" / "deck" / "ardour"
SESSION = """<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<Session version=\"3002\" name=\"juniordeck\" sample-rate=\"48000\">
  <Routes>
    <Route name=\"master\" default-type=\"audio\" channels=\"2\"/>
    <Route name=\"deck\" default-type=\"audio\" channels=\"2\"/>
  </Routes>
</Session>
"""
LUA = "ardour { [1] = function (n, ...) return 0 end }\n"
def write():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "juniordeck.ardour").write_text(SESSION, encoding="utf-8")
    (OUT / "juniordeck.lua").write_text(LUA, encoding="utf-8")
    body = {"host": "ardour", "license": "GPL-2+", "vendored": False,
            "binary": bool(shutil.which("ardour")), "live": False,
            "files": ["juniordeck.ardour", "juniordeck.lua"],
            "bind": "127.0.0.1", "model_pull": False}
    (OUT / "manifest.json").write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    return body
