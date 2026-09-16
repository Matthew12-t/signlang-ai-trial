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

Request-schema failures use the shared `400 INVALID_REQUEST` envelope, unsupported
languages use `422 UNSUPPORTED_LANGUAGE`, and requests above `GLOSS_MAX_TOKENS` use
`413 PAYLOAD_TOO_LARGE`. Every success and error response includes `X-Request-ID`.
Token labels must contain at least one Unicode letter or digit. Deterministic fallback
replaces underscores with spaces but preserves meaningful punctuation and Unicode, so
unsupported labels such as `C++` and `Café` are not silently erased.

## Gloss modes and Hugging Face

`GLOSS_MODE=template` is the default. It uses deterministic rules for known token
patterns and produces an `UNMATCHED_TEMPLATE` warning when no rule matches. It never
calls Hugging Face.

`GLOSS_MODE=qwen` sends the confirmed token labels without destructive canonicalization
to `LLM_MODEL` (default
`Qwen/Qwen3-4B`) through Hugging Face. It requires `HF_TOKEN` with inference permission;
`HF_PROVIDER=auto` selects an available provider. Hugging Face serverless/Inference
Providers free-tier credit is limited, not permanently free, and provider/model
availability can change.

The provider prompt always includes the exact response JSON Schema and a compact valid
object example. `HF_USE_STRUCTURED_OUTPUT=true` additionally sends that schema through
the provider-native `response_format` option when the selected provider supports it.
Empty or whitespace-only `HF_TOKEN` values are treated as absent.

Hosted output is accepted only after local schema, provenance, and text validation.
Text must be non-empty printable ASCII, contain an English word signal related to the
input or the service's small common-English vocabulary, avoid high-confidence foreign
phrases such as `Terima kasih`, and use a conservative plain-text punctuation set.
This lightweight check is intentionally not a general language detector: uncommon but
valid English can be rejected, while every uncertain result safely returns the
deterministic template with `LLM_FALLBACK_INVALID_RESPONSE`. Non-ASCII deterministic
fallback remains supported; the ASCII restriction applies only to hosted model output.

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

The smoke test passes only when the service accepts an actual `method=qwen` result with
unchanged provenance and no fallback warning. A deterministic fallback is a smoke-test
failure because it does not establish live model/provider compatibility.

See `.env.example` for the complete configuration reference, including shared STT, TTS,
Recall, and local sign-service settings.
