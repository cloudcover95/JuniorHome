# JuniorDeck audio → OSai

Drop WAV, MP3, FLAC, Ardour `.ardour`, or a sidecar `.txt` into `~/.juniorhome/deck/inbox`.

```bash
python3 scripts/deck_audio_prod.py
```

WAV uses stdlib `wave` (PCM16). MP3/FLAC need `ffmpeg` on PATH or the row stays `ffmpeg_absent`. Hours are capped at 3600 s per pass and strided so the process does not hold the whole file as floats.

Trit pack is an RMS envelope, not a transcript. Objectives come from a sidecar note (verbs: build, print, fix, commit, project) or the filename. The LLM ticket is `audio_digest.json`. `model_pull` is false. A local model may read that file later; this script does not call one.

DJI Mic Mini has no local API. Dump WAV from the case, then drop the file. No pairing, no MQTT.

Slate / GL.iNet / OpenVPN are not hosted here. `scripts/net_operator.py` only reports whether `tun0` exists.
