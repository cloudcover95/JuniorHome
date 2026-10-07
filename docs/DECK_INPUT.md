# Deck input

Drop a PCM16 WAV in ~/.juniorhome/deck/inbox, then:

```bash
python3 scripts/deck_input_prod.py
```

Reads 8 frames into the line_in trit row. No sound device is opened.
live is true only when a WAV was read. Empty inbox stays zeros and live false.
