import json
from gguf_schema import dump as schema
from junioros_prod import main as prod
from ledger_verify import verify
from quant_error import errors
from sandbox_suite import run as sandbox
from trit_on import schema as math_schema
def end():
    suite = sandbox()
    book = errors()
    return {"saas": "junioros-local", "sandbox_ok": suite.get("ok"),
            "benches_us": suite.get("benches_us"), "trit_on": math_schema(),
            "gguf_meta": schema(), "market": book.get("pnl_from_book"),
            "refetch": book.get("refetch", {}).get("ran"), "ledger": verify(),
            "prod_ok": prod().get("suite_ok")}
if __name__ == "__main__":
    print(json.dumps(end(), indent=2, default=str))
