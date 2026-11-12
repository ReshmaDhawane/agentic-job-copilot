"""FastAPI service: start a run, inspect it, approve it (human-in-the-loop)."""
from __future__ import annotations

import uuid

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .graph import build_graph

app = FastAPI(title="Agentic Job Copilot")
graph = build_graph()


class RunRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=64)
    role: str = Field(min_length=1, max_length=200)
    job_description: str = Field(min_length=20, max_length=20_000)
    resume: str = Field(min_length=20, max_length=50_000)


def _view(thread_id: str, values: dict, waiting: bool) -> dict:
    return {
        "thread_id": thread_id,
        "status": "awaiting_approval" if waiting else values.get("status"),
        "fit_score": values.get("fit_score"),
        "gaps": values.get("gaps", []),
        "draft": values.get("draft"),
        "trace": values.get("trace", []),
    }


@app.get("/health")
async def health() -> dict:
    return {"ok": True}


@app.post("/runs")
async def start_run(req: RunRequest) -> dict:
    thread_id = str(uuid.uuid4())
    cfg = {"configurable": {"thread_id": thread_id}}
    await graph.ainvoke(req.model_dump(), cfg)
    snap = await graph.aget_state(cfg)
    return _view(thread_id, snap.values, waiting=bool(snap.next))


@app.post("/runs/{thread_id}/approve")
async def approve(thread_id: str) -> dict:
    cfg = {"configurable": {"thread_id": thread_id}}
    snap = await graph.aget_state(cfg)
    if not snap.values:
        raise HTTPException(404, "unknown run")
    if not snap.next:
        raise HTTPException(409, "run is not waiting for approval")
    await graph.ainvoke(None, cfg)
    snap = await graph.aget_state(cfg)
    return _view(thread_id, snap.values, waiting=False)
