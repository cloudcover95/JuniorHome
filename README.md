# JuniorHome

**Current Ecosystem State**

**Added SecureDigitalCallHandler with anti-bot voice verification**

New module `src/juniorllm/telephony/secure_call_handler.py` implements secure digital/mobile calling:

- Calls always start **MUTED** upon acceptance.
- Real-time verification for "non-bot verbal reflection" (live human speech vs TTS/bot).
- Only unmutes for full communication once verified.
- Extensible voice verifier injection (can use local STT, VAD + liveness detection, or BitNet-based models).
- Clean state machine (muted → verifying → verified unmuted).
- Designed to integrate with SHEEP memory (call logging), JuniorHome orchestrator, and crispy-mouse for hardware input.

This adds a sovereign, privacy-first digital calling capability to the ecosystem while strongly protecting against bot/spam calls.