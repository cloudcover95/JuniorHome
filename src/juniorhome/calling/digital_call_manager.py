# path: src/juniorhome/calling/digital_call_manager.py

"""
DigitalCallManager

Extended to forward vision detection events (from VisionTextEngine) to JuniorMemSys
for long-term pattern storage alongside call data.
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
        self.inference_engine: Optional[Any] = None
        self.memory_system: Optional[Any] = None
        self.memsys_call_pattern_store: Optional[Any] = None  # JuniorMemSys integration

    def set_verification_function(self, fn: Callable[[bytes], bool]):
        self.verification_fn = fn

    def set_inference_engine(self, engine: Any):
        self.inference_engine = engine

    def set_memory_system(self, memory_system: Any):
        self.memory_system = memory_system

    def set_memsys_call_pattern_store(self, store: Any):
        """Connect to JuniorMemSys CallPatternStore for long-term vision + call pattern storage."""
        self.memsys_call_pattern_store = store

    def accept_call(self, call_id: str) -> bool:
        if self.current_call_id is not None:
            return False

        self.current_call_id = call_id
        self.muted = True
        self.verified_human = False
        self.call_start_time = time.time()

        print(f"[DigitalCall] Call {call_id} accepted - MUTED (BitNet vision ready)")

        if self.memory_system:
            try:
                self.memory_system.record_event({"type": "call_accepted", "call_id": call_id, "timestamp": self.call_start_time})
            except:
                pass

        return True

    def feed_audio_chunk(self, audio_chunk: bytes) -> bool:
        if self.verified_human or not self.current_call_id:
            return False

        verified = False

        if self.inference_engine is not None:
            try:
                features = self._extract_audio_features(audio_chunk)
                state = {"active_profile": "voice_verification", "performance": {}, "audio_features": features}
                updated = self.inference_engine.train_step(state, 1.0)
                metrics = self.inference_engine.evaluate(updated)
                if metrics.get("is_human", 0) > 0.5:
                    verified = True
            except Exception as e:
                print(f"[DigitalCall] Inference engine error: {e}")

        elif self.verification_fn:
            try:
                verified = self.verification_fn(audio_chunk)
            except:
                pass
        else:
            if self._simple_energy(audio_chunk) > 0.02:
                verified = True

        if verified:
            self.verified_human = True
            self.muted = False
            print(f"[DigitalCall] Human verified - UNMUTED {self.current_call_id}")

            if self.memsys_call_pattern_store:
                try:
                    self.memsys_call_pattern_store.store_call_event({
                        "call_id": self.current_call_id,
                        "type": "call_unmuted",
                        "timestamp": time.time(),
                        "is_human_verified": True
                    })
                except:
                    pass

            return True
        return False

    def feed_vision_detection(self, detection: Dict[str, Any]) -> None:
        """Forward vision tag detection (from VisionTextEngine) to JuniorMemSys for pattern learning."""
        if self.memsys_call_pattern_store:
            try:
                self.memsys_call_pattern_store.store_call_event({
                    "type": "vision_tag_detected",
                    "timestamp": time.time(),
                    "detected_tags": detection.get("detected_tags", []),
                    "zoom_level": detection.get("zoom_level", 1.0),
                    "is_zoomed": detection.get("zoom_level", 1.0) > 1.5
                })
            except Exception as e:
                print(f"[DigitalCall] MemSys vision forward error: {e}")

    # ... (rest of methods unchanged for brevity - _simple_energy, mute, end_call, etc. remain the same)

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
        if self.current_call_id and self.memsys_call_pattern_store:
            try:
                self.memsys_call_pattern_store.store_call_event({
                    "type": "call_ended",
                    "call_id": self.current_call_id,
                    "timestamp": time.time()
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
