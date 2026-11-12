"""LLM provider abstraction.

Supports OpenAI, Azure OpenAI and Anthropic through one async interface.
When no API key is configured, a deterministic `OfflineLLM` is used so the
whole agent graph, tests and evaluation run without network access.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Protocol

import httpx


@dataclass
class LLMResponse:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0


class LLM(Protocol):
    name: str

    async def complete(self, system: str, user: str, *, json_mode: bool = False) -> LLMResponse: ...


def _rough_tokens(text: str) -> int:
    return max(1, len(text) // 4)


@dataclass
class OpenAICompatibleLLM:
    """OpenAI or Azure OpenAI chat completions over HTTP."""

    model: str
    api_key: str
    base_url: str = "https://api.openai.com/v1"
    azure_api_version: str | None = None
    timeout_s: float = 30.0
    name: str = "openai"

    async def complete(self, system: str, user: str, *, json_mode: bool = False) -> LLMResponse:
        if self.azure_api_version:
            url = f"{self.base_url}/openai/deployments/{self.model}/chat/completions?api-version={self.azure_api_version}"
            headers = {"api-key": self.api_key}
        else:
            url = f"{self.base_url}/chat/completions"
            headers = {"Authorization": f"Bearer {self.api_key}"}
        body: dict = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0.2,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        start = time.perf_counter()
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            r = await client.post(url, headers=headers, json=body)
            r.raise_for_status()
            data = r.json()
        usage = data.get("usage", {})
        return LLMResponse(
            text=data["choices"][0]["message"]["content"],
            model=self.model,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            latency_ms=(time.perf_counter() - start) * 1000,
        )


@dataclass
class AnthropicLLM:
    model: str
    api_key: str
    timeout_s: float = 30.0
    name: str = "anthropic"

    async def complete(self, system: str, user: str, *, json_mode: bool = False) -> LLMResponse:
        if json_mode:
            user += "\n\nRespond with a single JSON object only."
        start = time.perf_counter()
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            r = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": self.api_key, "anthropic-version": "2023-06-01"},
                json={"model": self.model, "max_tokens": 1024, "system": system,
                      "messages": [{"role": "user", "content": user}]},
            )
            r.raise_for_status()
            data = r.json()
        usage = data.get("usage", {})
        return LLMResponse(
            text="".join(b.get("text", "") for b in data["content"]),
            model=self.model,
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            latency_ms=(time.perf_counter() - start) * 1000,
        )


@dataclass
class OfflineLLM:
    """Deterministic stand-in used for tests, CI and demos without API keys.

    It understands the three prompt "tasks" the agents send (plan, extract,
    write) and answers with rule-based output in the same format a real model
    would return, so the rest of the system is exercised end to end.
    """

    name: str = "offline"
    model: str = "offline-deterministic"
    calls: list[str] = field(default_factory=list)

    async def complete(self, system: str, user: str, *, json_mode: bool = False) -> LLMResponse:
        start = time.perf_counter()
        task = re.search(r"TASK:\s*(\w+)", system)
        task_name = task.group(1) if task else "chat"
        self.calls.append(task_name)
        if task_name == "plan":
            text = json.dumps({"steps": [
                {"id": 1, "action": "extract_requirements", "goal": "List the skills the job description asks for"},
                {"id": 2, "action": "retrieve_evidence", "goal": "Find resume evidence for each requirement"},
                {"id": 3, "action": "score_fit", "goal": "Score match and list gaps"},
                {"id": 4, "action": "draft_note", "goal": "Write a grounded application note"},
            ]})
        elif task_name == "extract":
            jd = user.split("JOB DESCRIPTION:", 1)[-1]
            text = json.dumps({"requirements": _keyword_requirements(jd)})
        elif task_name == "write":
            evidence = re.findall(r"- \[(.+?)\] (.+)", user)
            role = re.search(r"ROLE:\s*(.+)", user)
            lines = [f"I am applying for the {role.group(1).strip() if role else 'advertised'} role."]
            for req, ev in evidence[:4]:
                lines.append(f"For {req}: {ev.strip()}")
            text = "\n".join(lines)
        else:
            text = "OK"
        return LLMResponse(text=text, model=self.model, input_tokens=_rough_tokens(system + user),
                           output_tokens=_rough_tokens(text), latency_ms=(time.perf_counter() - start) * 1000)


KNOWN_SKILLS = [
    "python", "typescript", "langgraph", "langchain", "rag", "vector database", "faiss", "pinecone",
    "docker", "kubernetes", "ci/cd", "azure", "aws", "gcp", "fastapi", "microservices", "llm",
    "prompt engineering", "evaluation", "guardrails", "observability", "semantic kernel", "autogen",
    "crewai", "flutter", "react native", "kotlin", "swift", "sql", "redis", "message queue",
]


def _keyword_requirements(jd: str) -> list[str]:
    low = jd.lower()
    return [s for s in KNOWN_SKILLS if s in low]


def get_llm() -> LLM:
    """Pick a provider from environment variables; fall back to offline mode."""
    if key := os.getenv("AZURE_OPENAI_API_KEY"):
        return OpenAICompatibleLLM(model=os.environ["AZURE_OPENAI_DEPLOYMENT"], api_key=key,
                                   base_url=os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/"),
                                   azure_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01"),
                                   name="azure-openai")
    if key := os.getenv("OPENAI_API_KEY"):
        return OpenAICompatibleLLM(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), api_key=key)
    if key := os.getenv("ANTHROPIC_API_KEY"):
        return AnthropicLLM(model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-latest"), api_key=key)
    return OfflineLLM()
