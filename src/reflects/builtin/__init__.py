"""Built-in default Reflects + factory dispatch.

These wrap the existing in-tree ``Hypothalamus`` and
``IncrementalRecall`` factory as default-registered Reflects. The
``BUILTIN_FACTORIES`` dict maps the names users write in
``config.yaml``'s ``reflects:`` list to the constructors that turn
``RuntimeDeps`` into a concrete Reflect instance.

User-toggleable Reflects (Reflect #1, #2, #3 from the design doc)
land in sibling files here and add an entry to ``BUILTIN_FACTORIES``.
"""
from typing import Any, Callable

from src.reflects.builtin.default_hypothalamus import (  # noqa: F401
    DefaultHypothalamusReflect,
)
from src.reflects.builtin.default_recall_anchor import (  # noqa: F401
    DefaultRecallAnchorReflect,
)
from src.reflects.protocol import Reflect


# Name → factory function. The factory takes ``RuntimeDeps`` so each
# Reflect can grab whichever LLM / embedder / config it needs from
# the same dependency bundle Runtime uses. Kept as a plain dict so
# user-installed Reflects can extend it via ``register_builtin``
# without inheriting from anything.
BUILTIN_FACTORIES: dict[str, Callable[[Any], Reflect]] = {
    "default_hypothalamus": lambda deps: DefaultHypothalamusReflect(
        deps.hypo_llm
    ),
    "default_recall_anchor": lambda deps: DefaultRecallAnchorReflect(),
}


def register_builtin(name: str, factory: Callable[[Any], Reflect]) -> None:
    """Add a Reflect factory under ``name`` so ``config.yaml`` can
    reference it. Intended for in-tree built-ins; user-installed
    Reflects from a plugin folder will use a different mechanism
    (not yet implemented).
    """
    if name in BUILTIN_FACTORIES:
        raise ValueError(f"Reflect factory {name!r} already registered")
    BUILTIN_FACTORIES[name] = factory

