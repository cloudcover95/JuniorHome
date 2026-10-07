# JuniorDeck board class

Kept: audio drop, trit pack, 8x8 geode grid, gamma. Those paths are unchanged.

Added as class, live false until a node is present:
16 pads x 8 banks, 37 keys with aftertouch, pitch/mod wheels, 4 knobs, 1 encoder,
3 pedal jacks, stereo line in/out, headphone, 8 CV/gate, DIN MIDI, USB audio class, USB MIDI class, 7 inch display.

Excluded: vendor OS, Q-Link, Live control mode, vendor project import, quadrant pressure pads.
MPCe-style pads are patent pending (inMusic). Do not copy that geometry.
USB Audio Class and MIDI are published standards. License stays MIT.

```bash
python3 scripts/deck_board_prod.py
```
