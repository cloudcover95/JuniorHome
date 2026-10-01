# JuniorDeck tech sheet
Protocol goldend-osai-omega/1. Module cad.
Ports: 2.54 mm 2x8 + USB-C (audio/HID class).
Audio ingest: ~/.juniorhome/deck/inbox → audio_digest.json + gaia_mesh/deck_audio.jsonl.
WAV stdlib. MP3 via ffmpeg if present. Ardour track names via XML.
Objectives from sidecar text, not ASR. model_pull false. dji_api false.
Compute Pi 4/5 or SFF. 48 kHz when a class device is present AND gated.
Wave: clip on sin+drift → trit → sha3 ticket.
Mesh loopback only. Metric ms per note. ml_kem false. trit_mcu false.
OpenVPN/Slate/GL not hosted. tun0 is a probe only.
