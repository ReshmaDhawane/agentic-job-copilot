"""Guardrails: input screening, tool safety and output grounding."""
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from .retrieval import tokenize

INJECTION_PATTERNS = [
    r"ignore (all|any|the)? ?(previous|prior|above) instructions",
    r"disregard (the )?(system|previous) prompt",
    r"you are now",
    r"reveal (your|the) (system )?prompt",
    r"act as (an? )?(admin|developer|dan)",
    r"<\s*/?\s*system\s*>",
]
PII_PATTERNS = {
    "email": r"[\w.+-]+@[\w-]+\.[\w.]+",
    "phone": r"(?<!\d)(\+?\d[\d -]{8,}\d)(?!\d)",
    "aadhaar": r"\b\d{4}\s?\d{4}\s?\d{4}\b",
    "pan": r"\b[A-Z]{5}\d{4}[A-Z]\b",
}


def detect_injection(text: str) -> list[str]:
    low = text.lower()
    return [p for p in INJECTION_PATTERNS if re.search(p, low)]


def redact_pii(text: str) -> str:
    for name, pat in PII_PATTERNS.items():
        text = re.sub(pat, f"[{name.upper()}]", text)
    return text


STOP = set("a an the and or for of to in on with is are was i my as at by from this that be it".split())


def ungrounded_sentences(draft: str, evidence: list[str], min_support: float = 0.6) -> list[str]:
    """Return sentences whose content words are not supported by the evidence.

    A sentence passes when at least `min_support` of its content words appear in
    the retrieved resume chunks. This blocks the writer from inventing
    experience that the candidate does not have.
    """
    vocab = set(tokenize(" ".join(evidence)))
    bad = []
    for sent in re.split(r"(?<=[.!?])\s+|\n", draft):
        words = [w for w in tokenize(sent) if w not in STOP and len(w) > 2]
        if len(words) < 3 or sent.lower().startswith("i am applying"):
            continue
        support = sum(w in vocab for w in words) / len(words)
        if support < min_support:
            bad.append(sent.strip())
    return bad


@dataclass
class ToolCallRecord:
    tool: str
    ok: bool
    attempts: int
    error: str | None = None


@dataclass
class SafeToolRunner:
    """Runs tools with an allow-list, per-call timeout and bounded retries.

    Every call is recorded so the reliability of each tool can be scored and
    traced.
    """

    tools: dict[str, Callable[..., Awaitable[Any]]]
    timeout_s: float = 5.0
    max_retries: int = 2
    backoff_s: float = 0.05
    log: list[ToolCallRecord] = field(default_factory=list)

    async def run(self, name: str, **kwargs: Any) -> Any:
        if name not in self.tools:
            self.log.append(ToolCallRecord(name, False, 0, "tool not allowed"))
            raise PermissionError(f"tool '{name}' is not on the allow-list")
        last: Exception | None = None
        for attempt in range(1, self.max_retries + 2):
            try:
                result = await asyncio.wait_for(self.tools[name](**kwargs), self.timeout_s)
                self.log.append(ToolCallRecord(name, True, attempt))
                return result
            except Exception as exc:  # noqa: BLE001 - we record and retry any tool failure
                last = exc
                await asyncio.sleep(self.backoff_s * attempt)
        self.log.append(ToolCallRecord(name, False, self.max_retries + 1, repr(last)))
        raise RuntimeError(f"tool '{name}' failed after retries: {last!r}")

    def reliability(self) -> dict[str, float]:
        out: dict[str, list[int]] = {}
        for r in self.log:
            out.setdefault(r.tool, []).append(int(r.ok))
        return {k: sum(v) / len(v) for k, v in out.items()}
