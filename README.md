# JuniorHome

**Current Ecosystem State**

**Added Digital Calling + Mobile Integration**

New module `src/juniorhome/calling/digital_call_manager.py`:

- Full support for digital/mobile calling.
- **Always starts muted** on call acceptance.
- Stays muted until **verified non-bot human verbal speech** is detected.
- Once verified, automatically **unmutes** for complete two-way communication.
- Verification is fully pluggable — you can connect:
  - Simple VAD
  - BitNet / MLX voice models
  - Your black-box theoretical math engines
  - Advanced bot detection
- Optional integration with SHEEPMemory / JuniorMemSys for call event logging.
- Clean status API and force-mute / end-call controls.

This brings mobile calling into the sovereign edge orchestrator while maintaining strong privacy and anti-bot protection.

The feature is designed to work alongside the existing inference pipelines, plasticity, and memory systems.