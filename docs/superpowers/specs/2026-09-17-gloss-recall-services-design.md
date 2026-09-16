# Gloss Normalization and Conversation Recall Services Design

**Status:** Approved design  
**Date:** 2026-09-17  
**Scope:** Isyara AI Services MVP

## 1. Goals

Build two English-only AI services for the Isyara MVP:

1. Gloss Normalization converts confirmed sign tokens into natural English text.
2. Conversation Recall answers questions only from final transcript entries and returns verifiable evidence.

The initial LLM is Qwen served through Hugging Face Inference Providers. The application must keep the LLM boundary provider-neutral so another hosted API or a local model can replace Hugging Face without changing Gloss or Recall domain logic.

## 2. Delivery Strategy

Work is delivered in two sequential feature branches:

1. `feat/gloss-normalization-service`
2. Merge the Gloss branch into `main`.
3. Create `feat/conversation-recall-service` from the updated `main`.

The Gloss branch also establishes the shared FastAPI, configuration, error, data-model, observability, and chat-provider foundations required by both services.

## 3. Runtime Architecture

The MVP runs as one modular FastAPI application with independent routers and service layers:

```text
FastAPI application
├── shared
│   ├── configuration
│   ├── data models
│   ├── error handling
│   ├── observability
│   └── providers
│       ├── ChatProvider protocol
│       └── HuggingFaceChatProvider
├── gloss
│   ├── HTTP API
│   ├── application service
│   └── deterministic rules
└── recall
    ├── HTTP API
    ├── application service
    ├── prompt construction
    ├── context retrieval boundary
    └── evidence validation
```

Gloss and Recall have separate configuration and dependency boundaries so they can be deployed as separate processes later without changing their HTTP contracts.

The MVP does not use LangChain or a vector database. A small `ContextRetriever` protocol keeps future semantic retrieval or LangChain integration possible without coupling the current Recall service to a retrieval framework.

## 4. LLM Provider Boundary

Gloss and Recall depend on this provider-neutral behavior:

```python
class ChatProvider(Protocol):
    async def complete_json(
        self,
        messages: list[ChatMessage],
        response_schema: dict[str, object],
        *,
        max_tokens: int,
        temperature: float,
    ) -> dict[str, object]: ...
```

`HuggingFaceChatProvider` is the first implementation. It uses `huggingface_hub.InferenceClient` and Hugging Face Inference Providers with an HF token. Provider and model selection are configuration values, not domain constants.

Initial configuration:

```dotenv
LLM_BACKEND=huggingface
LLM_MODEL=Qwen/Qwen3-4B
HF_PROVIDER=auto
HF_TOKEN=
```

The initial `auto` provider policy allows the MVP to use whichever provider currently serves the configured Qwen model. A smoke test may later pin a provider when that improves structured-output compatibility or latency. Free Hugging Face usage is a limited free tier, not an unlimited zero-cost guarantee.

Provider-native structured output is used when available, but service correctness does not depend on it. Responses are always parsed and validated locally with Pydantic. Provider exceptions are translated into application errors without exposing credentials, provider URLs, or stack traces.

## 5. Gloss Normalization

### 5.1 Contract

The endpoint is `POST /v1/gloss/normalize`. It accepts an utterance ID, `language="en"`, and a non-empty ordered list of confirmed sign tokens. It returns normalized text, the method used, every source token ID, warnings, request ID, and latency.

Only confirmed tokens are valid inputs. The service is stateless and does not create sessions, utterances, or transcript entries.

### 5.2 Deterministic Default

`GLOSS_MODE=template` is the default. Processing is deterministic:

1. Normalize token-label spelling and separators.
2. Match the longest supported multi-token template.
3. Fall back to per-token vocabulary mappings and small English grammar rules.
4. Apply capitalization and punctuation.
5. Preserve unsupported labels safely and emit warnings instead of inventing meaning.

The same ordered token list always produces the same output. The initial exact templates are:

| Ordered labels | Output |
|---|---|
| `I NOT_UNDERSTAND` | `I do not understand.` |
| `YOU REPEAT PLEASE` | `Could you repeat that, please?` |
| `REPEAT PLEASE` | `Please repeat that.` |
| `THANK_YOU` | `Thank you.` |
| `HELLO` | `Hello.` |
| `YES` | `Yes.` |
| `NO` | `No.` |

Labels outside an exact template are normalized by replacing underscores with spaces, lowercasing ordinary words, joining them in input order, capitalizing the first character, and appending a period. Such output includes an `UNMATCHED_TEMPLATE` warning so it is never mistaken for grammar-aware normalization. Additional vocabulary rules can be added as data without changing the service interface.

### 5.3 Qwen Feature Flag

`GLOSS_MODE=qwen` enables Qwen-based normalization through `ChatProvider`. The prompt contains only the confirmed token labels and requests concise English text without adding facts.

The Qwen result is accepted only when:

- its schema is valid;
- its source token IDs exactly match the request;
- it returns non-empty English text;
- it does not report unsupported extra source tokens.

If inference times out, the provider is unavailable, or validation fails, Gloss returns the deterministic template result with a warning. Therefore, Gloss remains usable without Hugging Face connectivity or credits.

## 6. Conversation Recall

### 6.1 Contract

The endpoint is `POST /v1/recall/query`. It accepts a session ID, English query, explicit final `contextEntries`, and answer options. It returns an answer, grounded status, evidence, not-found reason, model metadata, request ID, and latency.

The Recall service is stateless. It never loads meeting history by session ID; the Application Server supplies the selected context on every request.

### 6.2 Context Handling

The MVP `ProvidedContextRetriever` receives the supplied transcript entries, orders them by sequence, and applies `RECALL_MAX_CONTEXT_CHARS=24000` by removing the oldest complete entries. It never truncates an entry in the middle. A single entry longer than 24,000 characters is rejected as `PAYLOAD_TOO_LARGE` instead of being partially included.

If context is empty, Recall returns `grounded=false` and `notFoundReason=EMPTY_CONTEXT` without calling the LLM.

The future `ContextRetriever` protocol may be implemented with embeddings, a vector store, or LangChain when transcript size requires semantic retrieval. That future implementation must return the same `TranscriptEntry` domain objects used by the current service.

### 6.3 Prompt and Output

The system prompt instructs Qwen to:

- answer only from supplied transcript entries;
- avoid external knowledge;
- return `grounded=false` when the answer is unsupported;
- reference existing entry IDs and quote their text;
- keep the answer concise and in English;
- return data matching the supplied schema.

Transcript entries are sent as structured JSON rather than an unlabelled concatenated string. The initial inference settings are temperature zero and a maximum of 160 answer tokens.

### 6.4 Evidence Validation

Every model response is validated after Pydantic parsing:

1. Each evidence entry ID must exist in the exact context sent to Qwen.
2. Each evidence quote must equal or be a normalized substring of the referenced transcript entry.
3. `grounded=true` requires at least one valid evidence item.
4. `grounded=false` must return an empty evidence list.

Validation failure produces a safe response with `grounded=false`, empty evidence, and `notFoundReason=EVIDENCE_VALIDATION_FAILED`. Qwen is never treated as the validator of its own output.

## 7. HTTP, Security, and Errors

The application exposes:

```text
POST /v1/gloss/normalize
POST /v1/recall/query
GET  /health/live
GET  /health/ready
```

`X-Request-ID` is accepted from the caller or generated by the application and returned on every response. `X-Internal-API-Key` is required when an internal key is configured. Language values other than `en` are rejected.

Errors use the shared envelope defined by the existing architecture document. Required mappings are:

| HTTP | Code | Meaning |
|---:|---|---|
| 400 | `INVALID_REQUEST` | Invalid schema or parameters |
| 413 | `PAYLOAD_TOO_LARGE` | Context exceeds an accepted hard limit |
| 422 | `UNSUPPORTED_LANGUAGE` | Language is not English |
| 429 | `RATE_LIMITED` | Provider or service rate limit |
| 502 | `UPSTREAM_BAD_RESPONSE` | Malformed or unusable provider response |
| 503 | `MODEL_UNAVAILABLE` | Configured model or provider unavailable |
| 504 | `HF_TIMEOUT` | Provider exceeded its deadline |

Logs include request ID, operation, configured model/provider, latency, status, input size, context-entry count, and token usage when available. Logs exclude secrets, raw media, complete prompts, and full transcript content by default.

`/health/live` checks only the process. `/health/ready` checks configuration and reports provider availability without exposing credentials. Gloss template mode remains ready even when the LLM provider is unavailable; Recall readiness reports degradation when it cannot use its configured provider.

## 8. Testing Strategy

Unit tests use fake `ChatProvider` implementations and consume no hosted inference credits.

Gloss tests cover:

- exact and multi-token template rules;
- deterministic output;
- unknown-token preservation and warnings;
- unsupported languages and empty tokens;
- Qwen mode through the provider protocol;
- fallback on timeout, provider failure, malformed JSON, and invalid token provenance.

Recall tests cover:

- prompt contents and absence of external context;
- empty-context short circuit;
- ordering and whole-entry context limits;
- valid evidence;
- unknown evidence entry IDs;
- quote mismatch;
- grounded answers without evidence;
- malformed provider output and mapped provider errors.

API and contract tests cover request and response schemas, request-ID propagation, internal API-key behavior, error envelopes, health endpoints, and OpenAPI paths.

Hugging Face smoke tests are opt-in and skipped unless `HF_TOKEN` and an explicit smoke-test flag are present. They verify that the configured model-provider combination responds and can produce locally validatable JSON.

## 9. Definition of Done

Each feature branch is complete when:

- its unit and contract tests pass locally;
- the service starts locally with documented commands;
- OpenAPI and README documentation match implemented behavior;
- configuration contains no committed credentials;
- responses and logs do not expose secrets;
- Gloss works in template mode without network access;
- Recall fails with a stable error contract when Qwen is unavailable;
- model and provider can be replaced through configuration and a new `ChatProvider` implementation without modifying Gloss or Recall domain logic.
