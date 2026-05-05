"""Core service slot resolver — turn config dotted paths into instances.

Reads ``Config.core_implementations``; for each slot the user fills,
imports the dotted path, instantiates with runtime-supplied kwargs,
and validates the result satisfies the slot's Protocol. Empty slots
fall back to ``default_factory(**kwargs)``.

Failure modes (all loud, no silent fallback to default — fail-fast at
startup beats failing 30 minutes into a session with a confusing
AttributeError):

  * malformed path: not ``module:Class`` → ``ValueError``
  * import fails: ``ImportError`` (annotated with the override path)
  * attribute missing on the imported module: ``ImportError``
  * instantiation TypeError on kwargs mismatch: ``TypeError``
    (annotated with the slot's expected kwarg list)
  * resulting object doesn't satisfy the Protocol: ``TypeError``
    listing missing attributes

All paths the user supplies are imported via ``importlib`` — there's no
allow-list. The user is opting in to running their own code by
declaring a dotted path; that's the same trust boundary as
``pip install`` of a package.
"""
from __future__ import annotations

import importlib
from typing import Any, Callable, TypeVar

from krakey.models.config import Config
from krakey.models.config.core_impls import CoreImplementations


T = TypeVar("T")
Importer = Callable[[str], Any]


def _default_importer(path: str) -> Any:
    """Resolve ``module.path:ClassName`` to the class object.

    Pure: never instantiates the class — the resolver does that with
    runtime-supplied kwargs after the import. Splitting "import" from
    "instantiate" lets tests stub one without the other.
    """
    module_path, _, attr = path.partition(":")
    if not attr:
        raise ValueError(
            f"core_implementations override path must be "
            f"'module.path:ClassName' (entry-point style), got {path!r}"
        )
    try:
        mod = importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(
            f"core_implementations: cannot import module "
            f"{module_path!r} from override path {path!r}: {e}"
        ) from e
    if not hasattr(mod, attr):
        raise ImportError(
            f"core_implementations: module {module_path!r} has no "
            f"attribute {attr!r} (referenced by override path {path!r})"
        )
    return getattr(mod, attr)


def _missing_protocol_attrs(instance: Any, protocol: type) -> list[str]:
    """List Protocol attributes the instance lacks.

    ``isinstance(obj, RuntimeCheckableProto)`` only tells you "no" —
    not WHICH methods are missing. This helper enumerates the public
    callable attributes of the Protocol and reports those absent on
    the instance, so the resolver can surface a useful error.
    """
    expected = {
        a for a in dir(protocol)
        if not a.startswith("_") and callable(getattr(protocol, a, None))
    }
    actual = set(dir(instance))
    return sorted(expected - actual)


class ServiceResolver:
    """Resolves a slot to either the user override or the built-in default.

    Constructed from a parsed ``Config``; one resolver per Runtime.
    The composition root keeps a reference and calls
    ``.resolve(slot, default_factory=..., expected_protocol=..., **kwargs)``
    once per swappable service during startup.
    """

    def __init__(
        self,
        cfg: Config,
        *,
        importer: Importer | None = None,
    ):
        self._overrides = cfg.core_implementations
        self._import = importer or _default_importer

    @classmethod
    def with_overrides(
        cls,
        overrides: dict[str, str],
        *,
        importer: Importer | None = None,
    ) -> "ServiceResolver":
        """Test-only constructor: build a resolver from a plain dict
        instead of a full Config object. Saves boilerplate in unit
        tests that only care about a handful of slots."""
        cfg = Config(core_implementations=CoreImplementations(**overrides))
        return cls(cfg, importer=importer)

    def resolve(
        self,
        slot: str,
        *,
        default_factory: Callable[..., T],
        expected_protocol: type,
        **kwargs: Any,
    ) -> T:
        """Return either the user override or ``default_factory(**kwargs)``.

        ``kwargs`` are forwarded to the constructor of either the user
        override OR the default factory — both must accept the same
        kwargs (this is part of each slot's contract; document it on
        the Protocol).
        """
        path = self._overrides.get(slot)
        if not path:
            return default_factory(**kwargs)

        cls = self._import(path)
        try:
            instance = cls(**kwargs)
        except TypeError as e:
            raise TypeError(
                f"core_implementations.{slot} = {path!r} could not be "
                f"instantiated with kwargs {sorted(kwargs)}: {e}. "
                f"Custom implementations of slot {slot!r} must accept "
                f"the same kwargs as the built-in default."
            ) from e

        if not isinstance(instance, expected_protocol):
            missing = _missing_protocol_attrs(instance, expected_protocol)
            raise TypeError(
                f"core_implementations.{slot} = {path!r} does not "
                f"satisfy {expected_protocol.__name__}; missing "
                f"attributes: {missing or '(unknown — '
                f'Protocol may have non-method requirements)'}"
            )
        return instance
