"""``dispatch`` Engine — run a DecisionResult's side-effects.

The slot's catalog of impls lives in ``meta.yaml`` next to this file.
The DispatchEngine Protocol lives at
``krakey.interfaces.engines.dispatch``.
"""
from krakey.engines.dispatch.default import LocalDispatchEngine

__all__ = ["LocalDispatchEngine"]
