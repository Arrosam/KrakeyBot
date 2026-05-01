"""Standalone onboarding wizard — generates config.yaml on first run.

Importable: ``from krakey.onboarding import run_wizard``
Runnable:   ``krakey onboard``  (or ``python -m krakey.onboarding``)

Decoupled from the runtime: the wizard never imports Runtime or any
plugin code. Plugin enumeration goes through the
``krakey.plugin_system.catalogue`` helper, which only reads
``meta.yaml`` files (no plugin module is imported).
Generates a ``Config`` dataclass and serializes it via ``dump_config``
— the same path tests use, so the written file always round-trips
through ``load_config`` cleanly.
"""
from krakey.onboarding.wizard import run_wizard


__all__ = ["run_wizard"]
