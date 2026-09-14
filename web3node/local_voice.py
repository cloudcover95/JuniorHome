from pathlib import Path
from fieldcore_bridge import pick_juniorllm_port
from home_terraform import inject
from second_brain import write_note
from trit3_align import write as trit3_write
OSS = {"stt": "faster-whisper MIT", "tts": "piper1-gpl", "livekit_local": "CoreWorxLab/local-livekit-plugins"}
def digest(utterance):
    tf = inject(utterance)
    return {"stt": "text-in", "llm": pick_juniorllm_port(utterance), "tf": tf.get("port"),
            "tts": "piper-hook", "hid": "crispy-mouse",
            "trit3_aligned_bytes": len(trit3_write([[0.2,-0.1,0.3],[0.0,0.4,-0.2]])),
            "xai_paid": False, "oss": OSS,
            "brain": str(write_note(Path(__file__).resolve().parent/"vault", f"# voice\n{utterance}\n"))}
