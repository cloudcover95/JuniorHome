# Local voice OSS

Do not clone livekit/agents into Home.
Learn:
- https://github.com/SYSTRAN/faster-whisper MIT STT
- https://github.com/OHF-Voice/piper1-gpl TTS (rhasspy/piper archived)
- https://github.com/CoreWorxLab/local-livekit-plugins FasterWhisper+Piper LiveKit plugins
- https://github.com/agjs/voicebox MIT OpenAI-compat local audio
- https://github.com/pygodzilla/stt2tts-mcp hot-swap engines

Pipeline: STT → JuniorOSai/terraform → Piper hook. crispy-mouse = HID.
`python3 web3node/local_voice.py`
