"""Service Protocols — the DIP boundary for the dashboard.

Routes depend on these Protocols, not on the Runtime god-object or
concrete implementations. ``app_factory.create_app`` wires concrete
adapters (most of which just delegate to Runtime) at construction
time.

This means:
  - routes are testable without a full Runtime (hand in a fake service);
  - swapping a backend (say, a different memory store) is a services-
    layer change — routes don't notice;
  - each Protocol is narrow (ISP): a route asks for exactly what it
    uses, so a memory-only route doesn't transitively depend on the
    plugin report surface.

Per Protocol lives in its own module (``events.py``, ``web_chat.py``,
``memory.py``, ``prompts.py``, ``plugins.py``, ``config.py``);
concrete adapters live in ``adapters.py``. Importers go to the
specific module they need — keeps each route's import list a clear
shopping list of what it actually depends on.
"""
