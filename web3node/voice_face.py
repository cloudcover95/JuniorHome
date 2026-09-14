from pathlib import Path
from fieldcore_bridge import pick_juniorllm_port
from home_terraform import inject
from palace_sis import palace
from second_brain import write_note
from trit3_hw import stack
def turn(text):
    tf = inject(text)
    return {"text": text, "port": pick_juniorllm_port(text), "tf": tf.get("port"),
            "palace": palace(text).get("math"), "trit3": stack().get("layout"),
            "tts": "hook", "xai_voice_ws": False,
            "brain": str(write_note(Path(__file__).resolve().parent / "vault", f"# voice\n{text}\n"))}
if __name__ == "__main__":
    import json
    print(json.dumps(turn("juniorosai field voice flagstaff"), indent=2))
