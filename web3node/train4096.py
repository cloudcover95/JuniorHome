import json
from pathlib import Path
def recipe():
    spec = {"shape": [4096, 4096], "dtype": "float16", "where": "T1_spark",
            "built": False, "surrogate_n": 32}
    path = Path(__file__).resolve().parent / "vault" / "train4096.json"
    path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    return spec | {"path": str(path)}
