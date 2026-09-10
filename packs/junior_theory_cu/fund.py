"""Member shares + proposals. Local units only."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from packs.junior_theory_cu.charter import CHARTER
from packs.junior_theory_cu.forum import Forum
from packs.junior_theory_cu.net import LocalNet


@dataclass
class Member:
    name: str
    shares: int = 1


@dataclass
class Proposal:
    pid: int
    author: str
    title: str
    yes: int = 0
    no: int = 0
    status: str = "open"


@dataclass
class Fund:
    root: Path
    members: dict[str, Member] = field(default_factory=dict)
    proposals: list[Proposal] = field(default_factory=list)
    treasury: int = 0

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.forum = Forum(self.root / "forum")
        self.net = LocalNet(self.root / "net")
        self._load()

    def _path(self) -> Path:
        return self.root / "fund.json"

    def _load(self) -> None:
        if not self._path().exists():
            return
        d = json.loads(self._path().read_text(encoding="utf-8"))
        self.treasury = d.get("treasury", 0)
        self.members = {k: Member(**v) for k, v in d.get("members", {}).items()}
        self.proposals = [Proposal(**p) for p in d.get("proposals", [])]

    def _save(self) -> None:
        self._path().write_text(
            json.dumps(
                {
                    "treasury": self.treasury,
                    "charter": CHARTER.strip(),
                    "members": {k: asdict(v) for k, v in self.members.items()},
                    "proposals": [asdict(p) for p in self.proposals],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def join(self, name: str) -> Member:
        if name not in self.members:
            self.members[name] = Member(name)
            self.treasury += 1
            self.forum.publish("member", name, f"joined with 1 share")
            self.net.mint(name, 1)
            self._save()
        return self.members[name]

    def propose(self, author: str, title: str) -> Proposal:
        self.join(author)
        p = Proposal(len(self.proposals), author, title)
        self.proposals.append(p)
        self.forum.publish("proposal", author, title, title)
        self._save()
        return p

    def vote(self, name: str, pid: int, choice: str) -> Proposal:
        self.join(name)
        p = self.proposals[pid]
        if choice == "yes":
            p.yes += self.members[name].shares
        else:
            p.no += self.members[name].shares
        if p.yes > p.no and p.yes >= max(1, len(self.members) // 2):
            p.status = "passed"
        self.forum.publish("vote", name, f"{choice} on {p.title}")
        self.net.receipt(name, f"vote:{pid}:{choice}")
        self._save()
        return p
