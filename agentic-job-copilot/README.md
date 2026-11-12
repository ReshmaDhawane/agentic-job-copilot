# Agentic Job Copilot

A LangGraph multi-step agent that reads a job description and a resume, plans the work,
finds evidence with hybrid RAG, scores the fit, drafts an application note, checks every
sentence against the evidence, and waits for a human to approve before finishing.

It is built so that the agent **cannot invent experience**: anything not supported by the
resume is listed as a gap and removed from the draft.

## Architecture

```
            ┌────────────┐  injection?  ┌──────┐
 request ──►│ guard_input│─────────────►│ END  │
            └─────┬──────┘              └──────┘
                  ▼
   recall (long-term episodic memory, per-user isolated)
                  ▼
   plan  (Plan-and-Execute: LLM returns ordered steps)
                  ▼
   extract (LLM → JSON requirements, keyword fallback on bad JSON)
                  ▼
   retrieve (tool: search_resume, run concurrently with asyncio.gather
             through SafeToolRunner: allow-list, timeout, retries, reliability score)
                  ▼
   score (fit score + gaps)
                  ▼
   write ◄──────────────┐  (only evidence lines, PII redacted)
                  ▼     │
   critic ──────────────┘  reflexion loop: ungrounded sentences → revise (max 2), then strip
                  ▼
   ⏸ human approval (LangGraph interrupt + checkpointer)
                  ▼
   finalize (save episode to long-term memory)
```

| Concern | Where |
|---|---|
| Agent graph, Plan-and-Execute, reflexion loop, HITL | `copilot/graph.py` |
| Hybrid search (BM25 + vectors, Reciprocal Rank Fusion), chunking with overlap | `copilot/retrieval.py` |
| Short-term (graph state) and long-term episodic memory | `copilot/memory.py` |
| Prompt-injection screen, PII redaction, grounding check, safe tool runner | `copilot/guardrails.py` |
| OpenAI / Azure OpenAI / Anthropic providers + offline deterministic model | `copilot/llm.py` |
| Async FastAPI service (`/runs`, `/runs/{id}/approve`) | `copilot/api.py` |
| Evaluation harness (success rate, hallucination rate, latency, tokens) | `eval/run_eval.py` |
| Tests, Docker, GitHub Actions CI | `tests/`, `Dockerfile`, `.github/workflows/ci.yml` |

Every node writes a trace event (latency, tokens, tool reliability) into the state, returned by the API.

## Run

```bash
pip install -r requirements.txt
pytest -q                      # 8 tests
python -m eval.run_eval        # evaluation report
uvicorn copilot.api:app --reload
```

Without an API key it runs on a deterministic offline model. To use a real model set one of:

```bash
export OPENAI_API_KEY=...            # optional OPENAI_MODEL
export AZURE_OPENAI_API_KEY=... AZURE_OPENAI_ENDPOINT=... AZURE_OPENAI_DEPLOYMENT=...
export ANTHROPIC_API_KEY=...
```

```bash
curl -X POST localhost:8000/runs -H 'content-type: application/json' \
  -d "{\"user_id\":\"me\",\"role\":\"Agentic AI Engineer\",\"job_description\":\"$(cat data/sample_jd.txt)\",\"resume\":\"$(cat data/sample_resume.txt)\"}"
curl -X POST localhost:8000/runs/<thread_id>/approve
```

Docker: `docker build -t agentic-job-copilot . && docker run -p 8000:8000 agentic-job-copilot`

## Ideas to extend

- Swap the TF-IDF/SVD embedder for Azure OpenAI embeddings and FAISS / Azure AI Search.
- Add LangSmith or OpenTelemetry export for the trace events.
- Add a Postgres checkpointer so approvals survive restarts.
- A/B test two writer prompts with `eval/run_eval.py`.
