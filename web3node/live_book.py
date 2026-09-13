"""Yahoo chart v8 C/H/L for ticker_universe. Cache vault/live_book.json."""
from __future__ import annotations
import json, time, urllib.request
from pathlib import Path
from typing import Any
import numpy as np
from ticker_universe import SECTOR, UNIVERSE
CACHE = Path(__file__).resolve().parent / "vault" / "live_book.json"
YAHOO = "https://query1.finance.yahoo.com/v8/finance/chart/{sym}?interval=1d&range=3mo"

def _fetch_symbol(sym: str) -> dict | None:
    req = urllib.request.Request(YAHOO.format(sym=sym), headers={"User-Agent": "JuniorHome/stocksnode"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None
    result = (payload.get("chart") or {}).get("result") or []
    if not result:
        return None
    row = result[0]
    quote = ((row.get("indicators") or {}).get("quote") or [{}])[0]
    out_t, out_c, out_h, out_l = [], [], [], []
    for t, c, h, l in zip(row.get("timestamp") or [], quote.get("close") or [], quote.get("high") or [], quote.get("low") or []):
        if c is None or h is None or l is None:
            continue
        out_t.append(int(t)); out_c.append(float(c)); out_h.append(float(h)); out_l.append(float(l))
    return {"t": out_t, "c": out_c, "h": out_h, "l": out_l} if len(out_t) >= 8 else None

def align(books: dict) -> dict[str, Any]:
    common = None
    for book in books.values():
        stamps = set(book["t"])
        common = stamps if common is None else common & stamps
    timeline = sorted(common)
    tickers = [name for name in UNIVERSE if name in books]
    close = np.zeros((len(tickers), len(timeline))); high = np.zeros_like(close); low = np.zeros_like(close)
    for i, name in enumerate(tickers):
        index = {t: j for j, t in enumerate(books[name]["t"])}
        for k, t in enumerate(timeline):
            j = index[t]
            close[i, k] = books[name]["c"][j]; high[i, k] = books[name]["h"][j]; low[i, k] = books[name]["l"][j]
    return {"tickers": tickers, "sectors": [SECTOR[name] for name in tickers], "timeline": timeline,
            "close": close, "high": high, "low": low, "source": "yahoo-chart-v8", "live": True}

def load_book(force: bool = False) -> dict[str, Any]:
    if CACHE.is_file() and not force:
        cached = json.loads(CACHE.read_text(encoding="utf-8"))
        if time.time() - float(cached.get("fetched_at", 0)) < 6 * 3600 and cached.get("live"):
            cached["close"] = np.asarray(cached["close"]); cached["high"] = np.asarray(cached["high"]); cached["low"] = np.asarray(cached["low"])
            return cached
    books = {}
    for name in UNIVERSE:
        row = _fetch_symbol(name)
        if row:
            books[name] = row
            time.sleep(0.15)
    if len(books) < 4:
        raise RuntimeError("live fetch too thin")
    aligned = align(books); aligned["fetched_at"] = time.time()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    serial = dict(aligned)
    serial["close"] = aligned["close"].tolist(); serial["high"] = aligned["high"].tolist(); serial["low"] = aligned["low"].tolist()
    CACHE.write_text(json.dumps(serial), encoding="utf-8")
    return aligned
