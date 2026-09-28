import math, struct
from pathlib import Path
from junior_gamma import gamma_j
def tone(hz=220.0, sec=0.2, sr=8000, g=0.3):
    n = int(sr * sec); pcm = bytearray()
    for i in range(n):
        s = math.sin(2 * math.pi * hz * i / sr) * max(0.0, min(g, 1.0))
        pcm += struct.pack("<h", int(s * 16000))
    return bytes(pcm)
def wav(pcm, sr=8000):
    return b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16) + b"data" + struct.pack("<I", len(pcm)) + pcm
def render(samples=None):
    g = gamma_j(samples or [0.2, 0.8, 0.1, 0.9])
    dest = Path(__file__).resolve().parent / "vault" / "deck_gamma.wav"
    dest.parent.mkdir(exist_ok=True)
    dest.write_bytes(wav(tone(g=min(g, 1.0))))
    return {"gamma_j": g, "wav_B": dest.stat().st_size, "path": str(dest), "synth": False, "mic": False}
