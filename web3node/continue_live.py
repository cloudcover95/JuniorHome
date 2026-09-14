import json
from gguf_read import read_ours
from gguf_i2s import write_i2s
from ledger_verify import verify
from missing_live import live as base
from ste_run import run as ste_run
def live():
    write_i2s([0.2, -0.4, 0.1, 0.0]*8)
    row = base()
    row["gguf_read"] = read_ours()
    row["ledger_ok"] = verify()
    row["ste_run"] = ste_run()
    return row
if __name__ == "__main__":
    print(json.dumps(live(), indent=2, default=str))
