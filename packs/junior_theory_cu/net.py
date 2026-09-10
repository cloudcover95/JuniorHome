"""Spin a local unit net. Prefers JuniorLLM bitnet_pq; else hash receipts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


class LocalNet:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.ledger = self.root / "units.jsonl"
        self.chain = None
        try:
            from bitnet_pq.chain import Chain

            self.chain = Chain()
            self.chain.genesis("treasury", [1, 0, -1, 1] * 8)
        except Exception:
            self.chain = None

    def _row(self, kind: str, name: str, extra: str) -> None:
        prev = "0" * 16
        if self.ledger.exists():
            lines = self.ledger.read_text(encoding="utf-8").splitlines()
            if lines:
                prev = json.loads(lines[-1]).get("hdr", prev)
        hdr = hashlib.sha256(f"{prev}:{kind}:{name}:{extra}".encode()).hexdigest()[:32]
        rec = {"kind": kind, "name": name, "extra": extra, "prev": prev, "hdr": hdr, "pq": bool(self.chain)}
        with self.ledger.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        if self.chain is not None:
            try:
                self.chain.tick("treasury")
            except Exception:
                pass

    def mint(self, name: str, shares: int) -> None:
        self._row("mint", name, str(shares))

    def receipt(self, name: str, extra: str) -> None:
        self._row("receipt", name, extra)
