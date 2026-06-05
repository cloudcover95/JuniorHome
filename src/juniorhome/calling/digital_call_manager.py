# path: src/juniorhome/calling/digital_call_manager.py

"""
DigitalCallManager

Sovereign edge digital calling + mobile integration for JuniorHome.

Core behavior:
- Always start calls muted.
- Stay muted until verified non-bot human verbal speech is detected.
- Once verified, unmute for full two-way communication.
- Designed for local processing (privacy/sovereignty).
- Verification is pluggable so you can use BitNet, theoretical math engines,
  or custom VAD + bot detection models.

Fits with the existing ecosystem (SHEEP memory, inference pipelines, crispy-mouse macros).
"""

import time
from typing import Any, Callable, Dict, Optional


class DigitalCallManager:
    def __init__(self, node_id: str = "default"):
        self.node_id = node_id
        self.muted: bool = True
        self.verified_human: bool = False
        self.current_call_id: Optional[str] = None
        self.call_start_time: Optional[float] = None

        # Pluggable verification function
        # Signature: verifier(audio_chunk: bytes) -> bool
        self.verification_fn: Optional[Callable[[bytes], bool]] = None

        # Optional integration with SHEEP memory or other components
        self.memory_system: Optional[Any] = None

    def set_verification_function(self, fn: Callable[[bytes], bool]):
        """Set custom verification logic (VAD + non-bot detection).

        Can use:
        - Simple energy-based VAD
        - BitNet/MLX voice model
        - Your theoretical black-box math engine
        - External library (webrtcvad, etc.)
        """
        self.verification_fn = fn

    def set_memory_system(self, memory_system: Any):
        """Optional: connect to SHEEPMemory or JuniorMemSys for call logging."""
        self.memory_system = memory_system

    def accept_call(self, call_id: str) -> bool:
        """Accept an incoming digital/mobile call.

        Always starts muted.
        Returns True if call accepted.
        """
        if self.current_call_id is not None:
            print("[DigitalCall] Already in a call. Rejecting new call.")
            return False

        self.current_call_id = call_id
        self.muted = True
        self.verified_human = False
        self.call_start_time = time.time()

        print(f"[DigitalCall] Call {call_id} accepted - STARTING MUTED")

        if self.memory_system:
            try:
                self.memory_system.record_event({
                    "type": "call_accepted",
                    "call_id": call_id,
                    "timestamp": self.call_start_time,
                    "muted": True
                })
            except:
                pass

        # Start background monitoring (in real impl this would be a thread/async task)
        self._start_verification_monitor()
        return True

    def _start_verification_monitor(self):
        """Background monitoring for human voice verification.

        In production this would listen to incoming audio stream.
        For now it's a hook that external code can call with audio chunks.
        """
        print("[DigitalCall] Verification monitor started (muted until human verified)")

    def feed_audio_chunk(self, audio_chunk: bytes) -> bool:
        """Feed incoming audio chunk for verification.

        Call this from your audio input pipeline (mobile/digital call audio).
        Returns True if verification just succeeded and we unmuted.
        """
        if self.verified_human or not self.current_call_id:
            return False

        verified = False

        if self.verification_fn:
            try:
                verified = self.verification_fn(audio_chunk)
            except Exception as e:
                print(f"[DigitalCall] Verification function error: {e}")
                verified = False
        else:
            # Default simple energy-based VAD (placeholder)
            # In real use replace with proper non-bot detection
            energy = self._simple_energy(audio_chunk)
            if energy > 0.02:  # tunable threshold
                verified = True

        if verified:
            self.verified_human = True
            self.muted = False
            print(f"[DigitalCall] Human voice verified - UNMUTING call {self.current_call_id}")

            if self.memory_system:
                try:
                    self.memory_system.record_event({
                        "type": "call_unmuted",
                        "call_id": self.current_call_id,
                        "timestamp": time.time()
                    })
                except:
                    pass

            return True

        return False

    def _simple_energy(self, audio_chunk: bytes) -> float:
        """Very basic energy calculation as default verifier."""
        if not audio_chunk:
            return 0.0
        # Treat bytes as signed 16-bit samples
        try:
            import struct
            samples = struct.unpack(f"{len(audio_chunk)//2}h", audio_chunk[:len(audio_chunk)//2*2])
            if not samples:
                return 0.0
            rms = (sum(x*x for x in samples) / len(samples)) ** 0.5
            return rms / 32768.0  # normalize
        except:
            return 0.0

    def mute(self):
        """Force mute (e.g. for privacy or bot suspicion)."""
        self.muted = True
        print(f"[DigitalCall] Call {self.current_call_id} forced muted")

    def end_call(self):
        """End the current call."""
        if self.current_call_id:
            print(f"[DigitalCall] Ending call {self.current_call_id}")
            if self.memory_system:
                try:
                    self.memory_system.record_event({
                        "type": "call_ended",
                        "call_id": self.current_call_id,
                        "duration": time.time() - (self.call_start_time or time.time())
                    })
                except:
                    pass

        self.current_call_id = None
        self.muted = True
        self.verified_human = False
        self.call_start_time = None

    def is_muted(self) -> bool:
        return self.muted

    def is_verified(self) -> bool:
        return self.verified_human

    def get_status(self) -> Dict[str, Any]:
        return {
            "call_id": self.current_call_id,
            "muted": self.muted,
            "verified_human": self.verified_human,
            "in_call": self.current_call_id is not None
        }
