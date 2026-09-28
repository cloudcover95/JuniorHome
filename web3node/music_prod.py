import json
from pathlib import Path
from music_session import session
from second_brain import write_note
def note_on(status, note, vel):
    if status & 0xF0 != 0x90 or vel == 0:
        return None
    return {"ch": status & 0x0F, "note": note, "vel": vel / 127.0}
def prod(hits=None):
    row = session("junior-deck")
    if hits:
        from music_deck import pad
        row["pad"] = pad(hits)
    vault = Path(__file__).resolve().parent / "vault"
    vault.mkdir(exist_ok=True)
    path = vault / "deck.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    row["jsonl"] = str(path)
    row["brain"] = str(write_note(vault, f"# JuniorDeck {row['name']}\n"))
    return row
