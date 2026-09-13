"""ProfitabilityIdentityMatrix as language-terraform. No terraform apply."""
from __future__ import annotations
import json
from pathlib import Path
from checks_balance import check
from home_terraform import inject

def _evaluate(ticker, q_mark, spot, liq, hold_days=90):
    try:
        import importlib
        eng = importlib.import_module("src.stocksnode.engines.profitability_matrix").ProfitabilityIdentityMatrix()
        return eng.evaluate_singularity(ticker, q_mark, spot, liq, hold_days)
    except Exception:
        fee = 0.008
        tax = 0.32
        gross_apy = max(q_mark * 12.0, 0.0)
        gross = spot * (gross_apy / 100.0) * (hold_days / 365.0)
        net = gross - spot * fee - max(gross - spot * fee, 0) * tax * 0.6
        net_apy = (net / spot) * (365.0 / hold_days) * 100.0 if spot else 0.0
        rec = "ISOLATE_AND_MONITOR"
        if net_apy > 3.5 and q_mark > 0.6:
            rec = "EXECUTE_YIELD_FARM"
        return {"ticker": ticker, "q_mark_trigger": round(q_mark, 4), "net_apy": round(net_apy, 2),
                "recommendation": rec, "engine": "home-twin"}

def layer():
    book = Path(__file__).resolve().parent / "vault" / "live_book.json"
    rows = []
    if book.is_file():
        raw = json.loads(book.read_text(encoding="utf-8"))
        for i, name in enumerate(raw.get("tickers") or []):
            close = raw.get("close") or []
            spot = float(close[i][-1]) if close else 0.0
            q = min(abs(spot) / (abs(spot) + 50.0), 0.99)
            rows.append(_evaluate(name, q, spot, 0.5))
    tf = inject("profitability terraform stock node")
    return {"terraform": {k: tf.get(k) for k in ("port", "ok")}, "balance": check("profitability terraform stock node"),
            "rows": rows[:10], "hashicorp": False}

if __name__ == "__main__":
    import json as _j
    print(_j.dumps(layer(), indent=2, default=str))
