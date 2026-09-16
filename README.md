# Isyara AI Services

Isyara AI Services provides sign-language support components. The implemented scope is
gloss normalization from confirmed English sign tokens into concise English text; the
repository also carries shared configuration for STT, TTS, Conversation Recall, and a
local sign service. The implementation plan is documented in
[`isyara-ai-services-design.md`](isyara-ai-services-design.md).

## Install, run, and test

```bash
python -m pip install -e ".[dev]"
python -m uvicorn src.main:app --reload
python -m pytest -m "not smoke"
```

The generated, machine-readable OpenAPI contract is
[`contracts/openapi.json`](contracts/openapi.json). It is produced from `src.main:app`;
[`contracts/openapi.yaml`](contracts/openapi.yaml) is only a human-readable pointer.

## HTTP API

- `GET /health/live` confirms the process is running.
- `GET /health/ready` reports configuration and gloss-provider readiness; Qwen mode
  without a token is reported as degraded.
- `POST /v1/gloss/normalize` normalizes confirmed sign tokens.

For example:

```bash
curl -X POST http://127.0.0.1:8000/v1/gloss/normalize \
  -H "Content-Type: application/json" \
  -H "X-Request-ID: request-123" \
  -d '{
    "utteranceId": "utterance-1",
    "language": "en",
    "tokens": [
      {"id": "tok-1", "label": "THANK_YOU", "confirmedAt": "2026-09-17T10:00:00Z"}
    ]
  }'
```

A successful response has this shape:

```json
{
  "requestId": "request-123",
  "utteranceId": "utterance-1",
  "text": "Thank you.",
  "method": "template",
  "sourceTokenIds": ["tok-1"],
  "warnings": [],
  "latencyMs": 0
}
```

`X-Request-ID` is optional; the service generates one when it is omitted or invalid.
Set `INTERNAL_API_KEY` to require the optional `X-Internal-API-Key` header on the gloss
endpoint. Do not place a real API key or Hugging Face token in repository files.

## Gloss modes and Hugging Face

`GLOSS_MODE=template` is the default. It uses deterministic rules for known token
patterns and produces an `UNMATCHED_TEMPLATE` warning when no rule matches. It never
calls Hugging Face.

`GLOSS_MODE=qwen` sends normalized token labels to `LLM_MODEL` (default
`Qwen/Qwen3-4B`) through Hugging Face. It requires `HF_TOKEN` with inference permission;
`HF_PROVIDER=auto` selects an available provider. Hugging Face serverless/Inference
Providers free-tier credit is limited, not permanently free, and provider/model
availability can change.

Qwen-mode fallback warnings are deliberately stable:

- `LLM_FALLBACK_PROVIDER_UNAVAILABLE`: no configured provider, provider unavailable, or
  an unclassified provider response.
- `LLM_FALLBACK_PROVIDER_TIMEOUT`: provider request timed out.
- `LLM_FALLBACK_RATE_LIMITED`: provider rate limited the request.
- `LLM_FALLBACK_INVALID_RESPONSE`: provider output did not validate.
- `LLM_FALLBACK_INVALID_PROVENANCE`: provider output did not preserve source token IDs.

Fallback results use deterministic templates. To add another LLM API, implement the
`ChatProvider` interface and wire the adapter into the application factory in
`src.main.create_app`.

Normal tests never call Hugging Face. The smoke test is intentionally opt-in and calls
the provider only when both `RUN_HF_SMOKE=1` and `HF_TOKEN` are present:

```bash
RUN_HF_SMOKE=1 HF_TOKEN=your_token python -m pytest -m smoke
```

See `.env.example` for the complete configuration reference, including shared STT, TTS,
Recall, and local sign-service settings.
