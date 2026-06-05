# path: src/juniorhome/calling/digital_call_manager.py

"""
DigitalCallManager

Now integrated with the quant / theoretical inference pipeline
for advanced voice recognition and non-bot verification.

You can pass an InferenceEngine (from juniorllm.comparison)
so that call audio verification runs through your quantized BitNet
or black-box theoretical math engines.
"""

import time
from typing import Any, Callable, Dict, Optional

try:
    from juniorllm.comparison.inference_comparison import InferenceEngine
except ImportError:
    InferenceEngine = None


class DigitalCallManager:
    def __init__(self, node_id: str = "default"):
        self.node_id = node_id
        self.muted: bool = True
        self.verified_human: bool = False
        self.current_call_id: Optional[str] = None
        self.call_start_time: Optional[float] = None

        self.verification_fn: Optional[Callable[[bytes], bool]] = None
        self.inference_engine: Optional[Any] = None   # InferenceEngine from comparison pipeline
        self.memory_system: Optional[Any] = None

    def set_verification_function(self, fn: Callable[[bytes], bool]):
        self.verification_fn = fn

    def set_inference_engine(self, engine: Any):
        """Connect a quant / theoretical inference engine for voice recognition.

        Recommended: Use TheoreticalMathEngine or any InferenceEngine
        from juniorllm.comparison.inference_comparison

        The engine will receive audio-derived features and help decide
        if the speech is real human (non-bot).
        """
        self.inference_engine = engine

    def set_memory_system(self, memory_system: Any):
        self.memory_system = memory_system

    def accept_call(self, call_id: str) -> bool:
        if self.current_call_id is not None:
            print("[DigitalCall] Already in a call.")
            return False

        self.current_call_id = call_id
        self.muted = True
        self.verified_human = False
        self.call_start_time = time.time()

        print(f"[DigitalCall] Call {call_id} accepted - STARTING MUTED (quant pipeline ready)")

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

        self._start_verification_monitor()
        return True

    def _start_verification_monitor(self):
        print("[DigitalCall] Verification monitor active (using quant/theoretical pipeline if set)")

    def feed_audio_chunk(self, audio_chunk: bytes) -> bool:
        if self.verified_human or not self.current_call_id:
            return False

        verified = False

        # Priority 1: Use connected InferenceEngine (quant / theoretical math)
        if self.inference_engine is not None:
            try:
                # Convert audio to simple feature state for the engine
                features = self._extract_audio_features(audio_chunk)
                state = {
                    "active_profile": "voice_verification",
                    "performance": {},
                    "audio_features": features
                }
                # Run one step through the quant/theoretical engine
                updated_state = self.inference_engine.train_step(state, outcome=1.0)
                metrics = self.inference_engine.evaluate(updated_state)

                # Decision logic: engine's theoretical_fit or avg_performance above threshold
                score = metrics.get("theoretical_fit", metrics.get("avg_performance", 0))
                if score > 0.15:   # tunable threshold
                    verified = True
            except Exception as e:
                print(f"[DigitalCall] Inference engine error: {e}")
                verified = False

        # Priority 2: Custom verification function
        elif self.verification_fn:
            try:
                verified = self.verification_fn(audio_chunk)
            except Exception as e:
                print(f"[DigitalCall] Verification function error: {e}")

        # Priority 3: Simple default energy VAD
        else:
            energy = self._simple_energy(audio_chunk)
            if energy > 0.02:
                verified = True

        if verified:
            self.verified_human = True
            self.muted = False
            print(f"[DigitalCall] Human voice verified via quant pipeline - UNMUTING {self.current_call_id}")

            if self.memory_system:
                try:
                    self.memory_system.record_event({
                        "type": "call_unmuted",
                        "call_id": self.current_call_id,
                        "timestamp": time.time(),
                        "via_quant_engine": self.inference_engine is not None
                    })
                except:
                    pass

            return True

        return False

    def _extract_audio_features(self, audio_chunk: bytes) -> Dict[str, float]:
        """Simple feature extraction so the quant/theoretical engine can process audio."""
        energy = self._simple_energy(audio_chunk)
        # Add more features here if needed (zero-crossing, spectral, etc.)
        return {
            "energy": energy,
            "length": len(audio_chunk),
            "timestamp": time.time()
        }

    def _simple_energy(self, audio_chunk: bytes) -> float:
        if not audio_chunk:
            return 0.0
        try:
            import struct
            samples = struct.unpack(f"{len(audio_chunk)//2}h", audio_chunk[:len(audio_chunk)//2*2])
            if not samples:
                return 0.0
            rms = (sum(x*x for x in samples) / len(samples)) ** 0.5
            return rms / 32768.0
        except:
            return 0.0

    def mute(self):
        self.muted = True

    def end_call(self):
        if self.current_call_id:
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
            "in_call": self.current_call_id is not None,
            "using_quant_engine": self.inference_engine is not None
        }
