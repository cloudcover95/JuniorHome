# ROM backend

Sniffs a file already in ~/.juniorhome/rom/inbox.
NES magic is NES\x1a. Other kinds come from the extension.
Looks for retroarch, sameboy, mgba, mednafen on PATH. Does not install them.
fetch false. bundled false. running false.

```bash
python3 scripts/rom_backend_prod.py
```
