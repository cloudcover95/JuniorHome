# JuniorDeck audio → OSai

Hardware is a USB sound-class device or a file on disk. No vendor stack.

Drop WAV, MP3, FLAC, an Ardour session, or a sidecar `.txt` into `~/.juniorhome/deck/inbox`.

```bash
python3 scripts/deck_audio_prod.py
```

WAV uses stdlib `wave` (PCM16). MP3/FLAC need `ffmpeg` on PATH or the row stays `ffmpeg_absent`. Hours are capped at 3600 s per pass and strided so the process does not hold the whole file as floats.

Trit pack is an RMS envelope, not a transcript. Objectives come from a sidecar note (verbs: build, print, fix, commit, project) or the filename. The LLM ticket is `audio_digest.json`. `model_pull` is false.
