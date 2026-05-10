"""DefaultContextEngine — Protocol conformance + identity vs. PromptBuilder.

The Engine is a subclass alias of PromptBuilder during the migration
window. Tests pin that identity contract so step 14's eventual file
move (krakey/prompt/ → krakey/engines/context/) doesn't accidentally
change call-site behavior."""
from __future__ import annotations

from krakey.engines.context.default import DefaultContextEngine
from krakey.interfaces.engines import ContextEngine
from krakey.prompt.builder import PromptBuilder


def test_satisfies_context_engine_protocol():
    eng = DefaultContextEngine()
    assert isinstance(eng, ContextEngine)


def test_subclasses_prompt_builder():
    """The default Engine must inherit PromptBuilder's behavior so
    every existing test that constructs PromptBuilder directly keeps
    working — it's the same surface."""
    assert issubclass(DefaultContextEngine, PromptBuilder)


def test_build_default_elements_round_trips():
    """Sanity: a full build_default_elements + render cycle produces
    a non-empty string. Doesn't pin layer order — that's covered in
    the dedicated PromptBuilder tests; this just verifies the Engine
    proxies through correctly."""
    from datetime import datetime
    from krakey.interfaces.engines.recall import RecallResult
    from krakey.prompt.views import StatusSnapshot

    eng = DefaultContextEngine()
    elements = eng.build_default_elements(
        self_model={"identity": {"name": "X", "persona": "Y"}},
        capabilities=[],
        status=StatusSnapshot(
            gm_node_count=0, gm_edge_count=0,
            fatigue_pct=0, fatigue_hint="",
            last_sleep_time="never", heartbeats_since_sleep=0,
        ),
        recall=RecallResult(),
        window=[],
        stimuli=[],
        current_time=datetime(2026, 5, 9, 12, 0, 0),
    )
    rendered = eng.render(elements)
    assert isinstance(rendered, str)
    assert len(rendered) > 0
