from pathlib import Path
def read_ours(path=None):
    path = path or Path(__file__).resolve().parent / "vault" / "juniorosai.i2s.gguf"
    raw = path.read_bytes()
    return {"magic": raw[:4].decode("ascii", "replace"), "bytes": len(raw),
            "ours": raw[:4] == b"GGUF", "llama_cpp_loadable": False}
