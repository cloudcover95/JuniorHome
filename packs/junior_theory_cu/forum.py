"""ForumMesh clone — append-only crowd events + gossip, no SQLAlchemy."""
from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

KINDS = ("general", "proposal", "vote", "treasury", "member", "help")


@dataclass
class CrowdEvent:
    event_id: str
    kind: str
    author: str
    title: str
    body: str
    origin_node: str = "local"
    created_at: str = ""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_event_id(author: str, body: str, created: str) -> str:
    return hashlib.sha256(f"{author}|{body}|{created}".encode()).hexdigest()[:32]


@dataclass
class Forum:
    root: Path
    events: list[CrowdEvent] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        p = self.root / "events.jsonl"
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self.events.append(CrowdEvent(**json.loads(line)))

    def publish(self, kind: str, author: str, body: str, title: str = "") -> CrowdEvent:
        kind = kind if kind in KINDS else "general"
        created = _now()
        ev = CrowdEvent(make_event_id(author, body, created), kind, author, title, body, "local", created)
        self.events.append(ev)
        with (self.root / "events.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(ev)) + "\n")
        return ev

    def export_bundle(self) -> dict:
        bundle = {
            "format": "junior-gossip-v1",
            "bundle_id": uuid.uuid4().hex[:16],
            "count": len(self.events),
            "events": [asdict(e) for e in self.events],
        }
        dest = self.root / "bundles"
        dest.mkdir(exist_ok=True)
        (dest / f"bundle_{bundle['bundle_id']}.json").write_text(json.dumps(bundle, indent=2), encoding="utf-8")
        return bundle
