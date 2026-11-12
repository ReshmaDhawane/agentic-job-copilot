"""Agent memory.

* Short-term memory lives in the LangGraph state (messages, plan, scratchpad).
* Long-term memory is an episodic store of past runs (JSONL on disk) with
  semantic lookup, so the planner can reuse lessons from similar jobs.
* Each user gets an isolated namespace: no cross-user reads.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .retrieval import tokenize


@dataclass
class Episode:
    user_id: str
    role: str
    fit_score: float
    gaps: list[str]
    lesson: str
    ts: float


class EpisodicMemory:
    def __init__(self, root: str | Path = ".memory"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _file(self, user_id: str) -> Path:
        safe = "".join(ch for ch in user_id if ch.isalnum() or ch in "-_") or "anon"
        return self.root / f"{safe}.jsonl"

    def add(self, ep: Episode) -> None:
        with self._file(ep.user_id).open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(ep)) + "\n")

    def all(self, user_id: str) -> list[Episode]:
        p = self._file(user_id)
        if not p.exists():
            return []
        return [Episode(**json.loads(line)) for line in p.read_text(encoding="utf-8").splitlines() if line]

    def recall(self, user_id: str, query: str, k: int = 2) -> list[Episode]:
        """Rank past episodes by token overlap with the new role (Jaccard)."""
        q = set(tokenize(query))
        scored = []
        for ep in self.all(user_id):
            t = set(tokenize(ep.role + " " + " ".join(ep.gaps)))
            if t and q:
                scored.append((len(q & t) / len(q | t), ep))
        scored.sort(key=lambda s: -s[0])
        return [ep for s, ep in scored[:k] if s > 0]


def new_episode(user_id: str, role: str, fit: float, gaps: list[str]) -> Episode:
    lesson = (f"Last time for '{role}', missing evidence for: {', '.join(gaps)}."
              if gaps else f"Strong match for '{role}'.")
    return Episode(user_id=user_id, role=role, fit_score=fit, gaps=gaps, lesson=lesson, ts=time.time())
