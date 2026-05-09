"""Bootstrap utilities — back-compat re-export shim (Engine refactor 2026-05).

Bootstrap behavior has moved into a self-contained plugin at
``krakey/plugins/bootstrap/``. This module keeps the old import paths
alive so existing tests / callers continue to work without churn:

  * ``krakey.bootstrap.BOOTSTRAP_PROMPT``               — plugin's prompt template
  * ``krakey.bootstrap.parse_self_model_update``        — plugin's NOTE parser
  * ``krakey.bootstrap.detect_bootstrap_complete``      — plugin's marker check
  * ``krakey.bootstrap.load_genesis``                   — plugin's GENESIS loader
  * ``krakey.bootstrap.load_self_model_or_default``     — moved to models.self_model

Direct imports of these symbols from this module keep working;
new code should import from the canonical locations:
  * ``krakey.plugins.bootstrap.prompt`` for BOOTSTRAP_PROMPT
  * ``krakey.plugins.bootstrap.state`` for the parsers + GENESIS loader
  * ``krakey.models.self_model`` for ``load_self_model_or_default``
"""
from __future__ import annotations

# Re-export from the new canonical locations.
from krakey.models.self_model import load_self_model_or_default  # noqa: F401
from krakey.plugins.bootstrap.prompt import BOOTSTRAP_PROMPT  # noqa: F401
from krakey.plugins.bootstrap.state import (  # noqa: F401
    detect_bootstrap_complete,
    load_genesis,
    parse_self_model_update,
)

__all__ = [
    "BOOTSTRAP_PROMPT",
    "detect_bootstrap_complete",
    "load_genesis",
    "load_self_model_or_default",
    "parse_self_model_update",
]
