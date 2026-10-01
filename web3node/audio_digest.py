"""JuniorDeck audio ingest. stdlib only. No DJI API. No model pull."""
from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
import time
import wave
import xml.etree.ElementTree as ET
from pathlib import Path

INBOX = Path.home() / ".juniorhome" / "deck" / "inbox"
MESH = Path.home() / ".juniorhome" / "gaia_mesh" / "deck_audio.jsonl"
TICKET = Path.home() / ".juniorhome" / "deck" / "audio_digest.json"
AUDIO = {".wav", ".aiff", ".aif", ".mp3", ".flac", ".ogg"}
DAW = {".ardour", ".lof", ".mid"}
NOTE = {".txt", ".md"}
VERBS = (
    "build", "print", "fix", "commit", "project", "objective",
    "shell", "pad", "deck", "terraform", "route",
)


def sha3_file(path: Path) -> str:
    h = hashlib.sha3_256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def clip(x: float) -> int:
    if x > 0.33:
        return 1
    if x < -0.33:
        return -1
    return 0


def _rms_trits(samples: list[float], n: int = 48) -> list[int]:
    if not samples:
        return [0] * n
    step = max(1, len(samples) // n)
    windows = []
    peak = 1.0
    for i in range(n):
        sl = samples[i * step:(i + 1) * step]
        if not sl:
            windows.append(0.0)
            continue
        rms = math.sqrt(sum(v * v for v in sl) / len(sl))
        windows.append(rms)
        peak = max(peak, rms)
    return [clip((rms / peak) * 2.0 - 1.0) for rms in windows]


def wav_pack(path: Path, cap_sec: int = 3600) -> dict:
    with wave.open(str(path), "rb") as w:
        ch = w.getnchannels()
        sw = w.getsampwidth()
        rate = w.getframerate() or 1
        frames = w.getnframes()
        take = min(frames, rate * cap_sec)
        raw = w.readframes(take)
    if sw != 2 or not raw:
        return {"ok": False, "reason": "need_pcm16", "rate": rate, "frames": frames}
    n = len(raw) // (2 * ch)
    samples = []
    stride = max(1, n // 48000)
    for i in range(0, n, stride):
        off = i * 2 * ch
        acc = 0
        for c in range(ch):
            acc += int.from_bytes(raw[off + 2 * c:off + 2 * c + 2], "little", signed=True)
        samples.append(acc / (ch * 32768.0))
    return {
        "ok": True,
        "rate": rate,
        "channels": ch,
        "sec": round(frames / rate, 2),
        "capped": frames > take,
        "trits": _rms_trits(samples),
    }


def ffmpeg_wav(src: Path, dest: Path) -> bool:
    ff = shutil.which("ffmpeg")
    if not ff:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [ff, "-y", "-i", str(src), "-ac", "1", "-ar", "16000", "-t", "3600", str(dest)],
        capture_output=True,
        check=False,
    )
    return proc.returncode == 0 and dest.exists()


def ardour_tracks(path: Path) -> list[str]:
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        return []
    names = []
    for el in root.iter():
        tag = el.tag.lower()
        if tag.endswith("route") or tag.endswith("track"):
            name = el.get("name")
            if name:
                names.append(name[:40])
    return names[:32]


def objectives(text: str) -> list[str]:
    hits = []
    for line in text.splitlines():
        low = line.lower()
        if any(v in low for v in VERBS) and len(line.strip()) > 8:
            hits.append(line.strip()[:160])
        if len(hits) >= 12:
            break
    return hits


def sidecar(path: Path) -> str:
    for ext in (".txt", ".md"):
        note = path.with_suffix(ext)
        if note.exists():
            return note.read_text(encoding="utf-8", errors="replace")[:20000]
    return ""


def digest_one(path: Path) -> dict:
    t0 = time.perf_counter()
    ext = path.suffix.lower()
    row = {
        "name": path.name,
        "sha3": sha3_file(path),
        "bytes": path.stat().st_size,
        "kind": ext,
        "dji": False,
        "pair": False,
        "live": False,
        "bind": "127.0.0.1",
    }
    note = sidecar(path)
    row["objectives"] = objectives(note or path.stem.replace("_", " "))
    if ext == ".wav":
        row["wave"] = wav_pack(path)
    elif ext in {".mp3", ".flac", ".ogg", ".aiff", ".aif"}:
        tmp = Path("/tmp") / f"jdeck_{row['sha3']}.wav"
        row["ffmpeg"] = ffmpeg_wav(path, tmp)
        row["wave"] = wav_pack(tmp) if row["ffmpeg"] else {"ok": False, "reason": "ffmpeg_absent"}
    elif ext == ".ardour":
        row["tracks"] = ardour_tracks(path)
    row["ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
    return row


def scan(inbox: Path | None = None) -> dict:
    box = inbox or INBOX
    box.mkdir(parents=True, exist_ok=True)
    MESH.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(box.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in AUDIO | DAW | NOTE:
            continue
        if path.suffix.lower() in NOTE and path.with_suffix(".wav").exists():
            continue
        rows.append(digest_one(path))
    ticket = {
        "protocol": "goldend-osai-omega/1",
        "module": "cad",
        "ask": "extract objectives from deck audio",
        "n": len(rows),
        "rows": rows,
        "llm": "ticket_only",
        "model_pull": False,
        "openvpn": False,
        "slate": False,
        "glinet": False,
        "dji_api": False,
    }
    TICKET.parent.mkdir(parents=True, exist_ok=True)
    TICKET.write_text(json.dumps(ticket, indent=2) + "\n", encoding="utf-8")
    with MESH.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps({"sha3": row["sha3"], "name": row["name"], "ms": row["ms"]}) + "\n")
    return ticket
