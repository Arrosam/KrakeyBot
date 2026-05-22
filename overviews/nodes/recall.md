# Node: recall

## Purpose
Per-beat memory recall: creates a short-lived RecallSession each beat. The session runs per-stimulus vector search against GM, optional rerank, weighted merge, token-budgeted selection, and neighbor expansion. RecallResult carries nodes, edges, covered_stimuli, uncovered_stimuli.

Optionally (default OFF), a **semantic-association layer** sends each stimulus's raw text to a bound LLM that derives multiple key-info phrases (key persons, dated events with relative dates resolved to ISO, places, relations, situational/role context, procedural/task context); each phrase becomes its own recall query through the existing pipeline. The layer degrades gracefully to the raw single-query behavior whenever it is disabled, the LLM is unbound/unavailable, or enrichment fails.

## Zone
memory-stack

## Implements contracts
- engine-protocols: RecallEngine Protocol
- recall-result: RecallResult dataclass, RecallSession lifecycle

## Depends on contracts
- engine-protocols: Implements RecallEngine
- recall-result: Defines RecallResult and RecallSession
- memory-access: Queries MemoryEngine for vector search + edges
- embedder-access: Uses embedder for stimulus-to-vector conversion
- llm-client: Optional — `ChatLike` via `LLMClientFactoryEngine.client_for_core_purpose`, used only when the semantic-association layer is enabled
- config-model: Recall tuning (per_stimulus_k, recall_token_budget, adrenalin weight) + engine config (semantic_association_enabled, semantic_association_purpose)

## Exposed interface
- `IncrementalRecallEngine(*, cfg, memory, embedder, reranker, factory=None, config=None)` — stateless factory; `factory`/`config` default to None so existing wiring is unaffected (the registry auto-injects `config`; `factory` is wired from the runtime)
- `IncrementalRecallEngine.new_session() → IncrementalRecall` — fresh per-beat session, optionally equipped with a `SemanticAssociationEnricher` when enabled + factory bound + purpose resolves
- `IncrementalRecall.add_stimuli(stimuli)` — 1..N queries per stimulus (raw text, or enriched phrases), per-stimulus vector search + scoring with intra-stimulus dedup
- `IncrementalRecall.finalize() → RecallResult` — token-budgeted selection + neighbor expansion (unchanged)

## Internal structure
- `default.py` — IncrementalRecallEngine factory; resolves the optional enricher from config + LLM factory on each `new_session()`
- `_internal/incremental.py` — IncrementalRecall session algorithm; `add_stimuli` multi-query loop with intra-stimulus weight dedup (cross-stimulus accumulation preserved)
- `_internal/enrich.py` — SemanticAssociationEnricher: thin async ChatLike wrapper; `enrich(text, *, now) → list[str] | None`, single LLM call, line-delimited phrase parsing, returns None on every error/empty condition
- `_internal/scoring.py` — Scripted reranking fallback formula

## Status
done

## Change log
- 2026-05-17: Indexed into modular-dev
- 2026-05-18: Added optional semantic-association layer — per-stimulus LLM keyword enrichment, default OFF (`semantic_association_enabled: false`). New `_internal/enrich.py`; `IncrementalRecall.add_stimuli` multi-query loop with intra-stimulus dedup; `IncrementalRecallEngine` accepts `factory`+`config` kwargs. New optional dependency on the `llm-client` contract. No behavior change when disabled.
- 2026-05-20: `semantic_association_purpose` default changed `"recall_enrichment"` → `"compact"`; `meta.yaml` schema token `text` → `core_purpose` (forward-looking annotation the dashboard consumes; meta_loader passes it through verbatim). `default.py` purpose resolution now collapses absent/null/empty/whitespace to `"compact"` via `(cfg.get(...) or "").strip() or "compact"` (non-empty values are stripped). Zero contract impact; no behavior change when `semantic_association_enabled` is false. Contract `recall-result` now `tested` (tests/recall-result.test.py, 15 edge tests).
- 2026-05-22: Enricher prompt expanded with two new phrase types: situational/role context (surfaces behavioral rules when Self is asked to act) and procedural/task context (surfaces past experience for actionable tasks). Example updated to Auckland ticket scenario demonstrating all six phrase types. Prompt-only change; no interface, config, or contract impact.
