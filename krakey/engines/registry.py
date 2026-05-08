"""``EngineRegistry`` — turn ``cfg.core_implementations.<slot>`` dotted
paths (or built-in defaults) into concrete Engine instances.

Replaces the previous ``ServiceResolver`` mechanism with the same
fail-fast semantics but a different scope: ServiceResolver was a
generic single-slot resolver; EngineRegistry is the Engine-only
resolver that ultimately produces a fully-populated ``EngineBundle``
in one batch (``resolve_all`` lands in step 12 once every default
impl exists).

Failure modes (all loud — DIP says fail-fast at startup beats failing
30 minutes into a session with a confusing AttributeError):

  * malformed path: not ``module:Class`` → ``ValueError``
  * import fails: ``ImportError`` annotated with the override path
  * attribute missing on the imported module: ``ImportError``
  * instantiation TypeError on kwargs mismatch: ``TypeError`` annotated
    with the slot name + the kwargs the caller supplied
  * resulting object doesn't satisfy the Protocol: ``TypeError``
    listing the missing attributes

All paths the user supplies are imported via ``importlib`` — there's
no allow-list. The user opting in to running their own code by
declaring a dotted path is the same trust boundary as ``pip install``
of a package.
"""
from __future__ import annotations

import importlib
from typing import Any, Callable, TypeVar

from krakey.models.config import Config

T = TypeVar("T")
Importer = Callable[[str], Any]


def _default_importer(path: str) -> Any:
    """Resolve ``module.path:ClassName`` to the class object.

    Pure: never instantiates the class — the registry does that with
    runtime-supplied kwargs after the import. Splitting "import" from
    "instantiate" lets tests stub one without the other.
    """
    module_path, _, attr = path.partition(":")
    if not attr:
        raise ValueError(
            f"engine override path must be 'module.path:ClassName' "
            f"(entry-point style), got {path!r}"
        )
    try:
        mod = importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(
            f"engines: cannot import module {module_path!r} "
            f"from override path {path!r}: {e}"
        ) from e
    if not hasattr(mod, attr):
        raise ImportError(
            f"engines: module {module_path!r} has no attribute "
            f"{attr!r} (referenced by override path {path!r})"
        )
    return getattr(mod, attr)


def _missing_protocol_attrs(instance: Any, protocol: type) -> list[str]:
    """List Protocol attributes the instance lacks.

    ``isinstance(obj, RuntimeCheckableProto)`` only tells you "no" —
    not WHICH methods are missing. This helper enumerates the public
    callable attributes of the Protocol and reports those absent on
    the instance, so the registry can surface a useful error.
    """
    expected = {
        a for a in dir(protocol)
        if not a.startswith("_") and callable(getattr(protocol, a, None))
    }
    actual = set(dir(instance))
    return sorted(expected - actual)


class EngineRegistry:
    """Resolves Engine slots to concrete instances.

    Constructed once per Runtime from a parsed ``Config``. Step 12
    will add a ``resolve_all(deps)`` batch method that walks every
    Engine slot in dependency order and returns an ``EngineBundle``;
    until then, callers use ``resolve(slot, default_path=..., ...)``
    one slot at a time.

    The user override is read from ``cfg.core_implementations.<slot>``
    via ``CoreImplementations.get(slot)``, which returns ``""`` for
    unset / unknown slots. Empty override falls back to the supplied
    ``default_path``. Both empty → ``ValueError`` (a slot with neither
    impl nor default is a runtime-killer; surface it loudly).
    """

    def __init__(
        self,
        cfg: Config,
        *,
        importer: Importer | None = None,
    ):
        self._cfg = cfg
        self._import = importer or _default_importer

    def resolve(
        self,
        slot: str,
        *,
        default_path: str,
        expected_protocol: type,
        **kwargs: Any,
    ) -> Any:
        """Return the user override or the default, instantiated.

        ``kwargs`` are forwarded to the constructor of either path —
        both must accept the same kwargs (this is part of each slot's
        contract; document on the Protocol).
        """
        override = self._cfg.core_implementations.get(slot)
        path = override or default_path
        if not path:
            raise ValueError(
                f"engine slot {slot!r}: no impl path "
                "(neither user override nor built-in default supplied). "
                "This is a runtime-killer — every Engine slot needs an "
                "impl before the runtime can start."
            )

        cls = self._import(path)
        try:
            instance = cls(**kwargs)
        except TypeError as e:
            raise TypeError(
                f"engine slot {slot!r} = {path!r} could not be "
                f"instantiated with kwargs {sorted(kwargs)}: {e}. "
                f"Custom engines for slot {slot!r} must accept the "
                "same kwargs as the built-in default."
            ) from e

        if not isinstance(instance, expected_protocol):
            missing = _missing_protocol_attrs(instance, expected_protocol)
            if missing:
                detail = f"missing attributes: {missing}"
            else:
                detail = (
                    "(Protocol may have non-method requirements that "
                    "isinstance cannot verify)"
                )
            raise TypeError(
                f"engine slot {slot!r} = {path!r} does not satisfy "
                f"{expected_protocol.__name__}; {detail}"
            )
        return instance
