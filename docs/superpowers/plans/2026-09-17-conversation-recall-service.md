# Conversation Recall Service Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a stateless, English-only Recall endpoint that answers only from supplied final transcript entries and returns locally validated evidence.

**Architecture:** A `ProvidedContextRetriever` sorts and trims whole transcript entries. `RecallService` builds a structured prompt, calls the existing provider-neutral `ChatProvider`, validates evidence locally, and exposes the result through the shared FastAPI application.

**Tech Stack:** Python 3.10+, FastAPI, Pydantic v2, `huggingface_hub`, pytest/pytest-asyncio.

**Spec:** `docs/superpowers/specs/2026-09-17-gloss-recall-services-design.md`

## Global Constraints

- English only; endpoint is `POST /v1/recall/query`.
- No LangChain or vector database in the MVP.
- The service is stateless and never loads history by `sessionId`.
- Normal tests must not call Hugging Face or consume credits.
- `grounded=true` always requires locally verified evidence.
- Logs and errors must not expose transcript text, prompts, tokens, or secrets.

---

### Task 1: Recall Models and Provided Context Retrieval

**Files:**
- Modify: `src/shared/models.py`
- Create: `src/recall/__init__.py`
- Create: `src/recall/retrieval.py`
- Test: `tests/unit/test_recall_retrieval.py`

**Interfaces:**
- Produces: `TranscriptEntry`, `RecallOptions`, `RecallQueryRequest`, `Evidence`, `LLMRecallResult`, `RecallQueryResponse`.
- Produces: `ContextRetriever.retrieve(entries, max_chars)` and `ProvidedContextRetriever`.

- [ ] Write failing tests proving entries are ordered by `sequence`, oldest whole entries are removed until the context fits, and a single oversized entry raises `RecallValidationError("PAYLOAD_TOO_LARGE")`.
- [ ] Run `python -m pytest tests/unit/test_recall_retrieval.py -v` and confirm failure because Recall models/retriever do not exist.
- [ ] Implement the models with the contract field aliases and a retriever that counts serialized entry text, never truncates an entry, and returns an empty list unchanged.
- [ ] Re-run the focused test and confirm it passes.
- [ ] Commit with `feat: add recall context retrieval`.

### Task 2: Prompt, Evidence Validation, and Recall Service

**Files:**
- Create: `src/recall/prompt.py`
- Create: `src/recall/validation.py`
- Create: `src/recall/service.py`
- Test: `tests/unit/test_recall_service.py`

**Interfaces:**
- Consumes: `ChatProvider.complete_json(...)`, Recall models, and `ContextRetriever`.
- Produces: `build_recall_messages(request, entries)` and `RecallService.query(request)`.

- [ ] Write failing tests for structured prompt content, empty-context short circuit, valid grounded evidence, `NOT_IN_CONTEXT`, unknown evidence IDs, quote mismatch, grounded-without-evidence, and malformed/provider error mapping.
- [ ] Run `python -m pytest tests/unit/test_recall_service.py -v` and confirm the expected missing-module failures.
- [ ] Implement a prompt containing only the query/options/selected entries plus the exact JSON schema. Send temperature `0` and the requested/configured token limit.
- [ ] Implement normalized substring evidence checks. Invalid evidence returns HTTP-200-safe `grounded=false`, empty evidence, and `EVIDENCE_VALIDATION_FAILED`; unsupported answers return `NOT_IN_CONTEXT`.
- [ ] Map provider failures to stable shared errors: rate limit `429`, bad response `502`, unavailable `503`, timeout `504`.
- [ ] Re-run the focused test and confirm it passes.
- [ ] Commit with `feat: add grounded recall service`.

### Task 3: Configuration and HTTP Integration

**Files:**
- Modify: `src/shared/config.py`
- Modify: `src/shared/providers/huggingface.py`
- Create: `src/recall/api.py`
- Modify: `src/main.py`
- Modify: `.env.example`
- Test: `tests/contract/test_recall_api.py`

**Interfaces:**
- Adds: `HF_RECALL_MODEL`, `RECALL_MAX_CONTEXT_CHARS`, `RECALL_MAX_ANSWER_TOKENS`, `RECALL_TEMPERATURE`.
- Adds: `POST /v1/recall/query` with the shared request-ID, auth, and error envelope.

- [ ] Write failing contract tests for success, empty context, unsupported language, invalid request, auth-before-body, provider errors, request-ID propagation, readiness, and OpenAPI.
- [ ] Run `python -m pytest tests/contract/test_recall_api.py -v` and confirm the route/configuration is missing.
- [ ] Extend the HF provider factory with an optional model override; construct Recall independently from Gloss configuration.
- [ ] Add the router and extend pre-body authentication to Recall. Readiness is degraded when Recall is enabled without a usable HF token.
- [ ] Re-run the contract tests and then `python -m pytest -m "not smoke"`.
- [ ] Commit with `feat: expose conversation recall API`.

### Task 4: Contract, Documentation, and Verification

**Files:**
- Modify: `README.md`
- Modify: `contracts/openapi.json`
- Modify: `contracts/openapi.yaml`
- Create: `tests/smoke/test_huggingface_recall.py`

**Interfaces:**
- Documents the runtime contract and an opt-in live compatibility test.

- [ ] Add an opt-in smoke test gated by both `RUN_HF_RECALL_SMOKE=1` and `HF_TOKEN`; it must require a valid, locally verified response and never run ordinarily.
- [ ] Document configuration, request/response examples, context limits, evidence rules, provider replacement, and limited HF credits.
- [ ] Regenerate `contracts/openapi.json` from `src.main:app` and verify semantic equality plus the Recall path.
- [ ] Run `python -m pytest -m "not smoke"`, `python -m compileall -q src tests`, `git diff --check`, and a secret scan.
- [ ] Commit with `docs: publish recall service contract`.

