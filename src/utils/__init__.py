"""Cross-cutting utilities (token counting, model metadata lookup).

Kept deliberately small so it doesn't accrete a junk drawer. If a helper
belongs to a specific subsystem (memory / prompt / plugins / ...),
it lives there; things here are only the ones that need to be reachable
from two or more subsystems without dragging a dependency across them.
"""
