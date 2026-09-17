# Task 2 Report: Gloss Models and Deterministic Rules

## Implementation summary

Implemented the shared Pydantic API/domain models and deterministic gloss normalization rules. Labels are canonicalized (trimmed, uppercased, and normalized to underscore-separated tokens), exact MVP templates are supported, and unmatched input receives a safe sentence-cased fallback with an `UNMATCHED_TEMPLATE` warning.

## Files changed

- `src/shared/models.py` — added `APIModel`, `ChatMessage`, `ConfirmedSignToken`, `GlossNormalizeRequest`, `GlossNormalizeResponse`, `LLMGlossResult`, and `RuleNormalization`.
- `src/gloss/rules.py` — added canonicalization, exact templates, and fallback normalization.
- `tests/unit/test_gloss_rules.py` — added exact-template, canonicalization, and fallback tests.

## TDD evidence

RED command:

```text
python -m pytest tests/unit/test_gloss_rules.py -v
...
collected 0 items / 1 error
ModuleNotFoundError: No module named 'src.gloss.rules'
exit_code=1
```

The initial failure was the expected missing production module during test collection.

GREEN command:

```text
python -m pytest tests/unit/test_gloss_rules.py -v
...
collected 9 items
tests\\unit\\test_gloss_rules.py ......... [100%]
9 passed in 0.33s
exit_code=0
```

## Full-suite result

```text
python -m pytest -v
...
collected 12 items
tests\\contract\\test_health.py ... [25%]
tests\\unit\\test_gloss_rules.py ......... [100%]
12 passed in 1.09s
exit_code=0
```

## Self-review

- Confirmed internal fields use snake_case and external aliases use camelCase where specified.
- Confirmed deterministic rules have no network/provider dependency.
- Confirmed source token IDs are retained in rule output.
- Confirmed exact templates emit no warnings and fallback emits `UNMATCHED_TEMPLATE`.
- Added the `sourceTokenIds` alias to `RuleNormalization` so its snake_case internal field can be constructed through the external alias, consistent with the project-wide alias constraint and the rule implementation.

## Concerns

Pytest emits an existing `pytest_asyncio` deprecation warning because `asyncio_default_fixture_loop_scope` is unset; this is unrelated to Task 2 and was not changed.

## Fix round 1

### Files changed

- `src/shared/models.py` — restricted `GlossNormalizeRequest.language` to `Literal["en"]`.
- `tests/unit/test_gloss_rules.py` — added a Pydantic `ValidationError` rejection test for `language="id"`.

### RED evidence

Before the model change:

```text
python -m pytest tests/unit/test_gloss_rules.py -v
...
collected 10 items
tests\\unit\\test_gloss_rules.py .........F [100%]
Failed: DID NOT RAISE <class 'pydantic_core._pydantic_core.ValidationError'>
1 failed, 9 passed in 0.48s
exit_code=1
```

### GREEN / covering test evidence

```text
python -m pytest tests/unit/test_gloss_rules.py -v
...
collected 10 items
tests\\unit\\test_gloss_rules.py .......... [100%]
10 passed in 0.39s
exit_code=0
```

### Full-suite evidence

```text
python -m pytest -v
...
collected 13 items
tests\\contract\\test_health.py ... [23%]
tests\\unit\\test_gloss_rules.py .......... [100%]
13 passed in 1.00s
exit_code=0
```

### Self-review

The request model now rejects every language other than exactly `"en"` through Pydantic validation; no other model or normalization behavior was changed. The existing pytest-asyncio deprecation warning remains unrelated.
