"""Standalone onboarding wizard — generates config.yaml on first run.

Importable: ``from src.onboarding import run_wizard``
Runnable:   ``python -m src.onboarding``

Decoupled from the runtime: the wizard never imports Runtime or any
plugin code beyond meta.yaml parsing (via ``plugin_catalogue``).
Generates a ``Config`` dataclass and serializes it via ``dump_config``
— the same path tests use, so the written file always round-trips
through ``load_config`` cleanly.
"""
from src.onboarding.wizard import run_wizard


__all__ = ["run_wizard"]
