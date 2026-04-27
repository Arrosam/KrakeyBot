"""Service Protocols — the DIP boundary for the dashboard.

Routes depend on these Protocols, not on the Runtime god-object or
concrete implementations. `app_factory.py` wires concrete adapters
(most of which just delegate to Runtime) at construction time.

This means:
  - routes are testable without a full Runtime (hand in a fake service);
  - swapping a backend (say, a different memory store) is a services-
    layer change — routes don't notice;
  - each Protocol is narrow (ISP): a route asks for exactly what it
    uses, so a memory-only route doesn't transitively depend on the
    plugin report surface.

Concrete adapters live in `services/adapters.py`. Keep Protocols free
of import-from-Runtime so tests can import them cheaply.
"""
from src.plugins.dashboard.services.events import EventBroadcasterService
from src.plugins.dashboard.services.web_chat import WebChatService
from src.plugins.dashboard.services.memory import MemoryService
from src.plugins.dashboard.services.prompts import PromptsService
from src.plugins.dashboard.services.plugins import PluginsService
from src.plugins.dashboard.services.config import ConfigService

__all__ = [
    "EventBroadcasterService",
    "WebChatService",
    "MemoryService",
    "PromptsService",
    "PluginsService",
    "ConfigService",
]
