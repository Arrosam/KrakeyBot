"""Built-in `web_chat` plugin — dashboard chat bi-directional channel.

Multi-component project: sensory (user → Krakey) + reply tentacle
(Krakey → user) share one `WebChatHistory` instance (built by Runtime
before plugin discovery because the dashboard WebSocket also
subscribes to it for broadcasting). WebChatHistory itself lives in
`src.dashboard.web_chat` — it's a data layer the dashboard owns, not
a plugin implementation detail.

Disable this project to silence chat IO while keeping the dashboard
up for monitoring — the sensory unregisters (no stimuli produced)
and the reply tentacle unregisters (Hypothalamus can't dispatch it).
Typed messages in the web UI are dropped with a log warning.

Layout within this package:
  - sensory.py  — WebChatSensory (user → Krakey)
  - tentacle.py — WebChatTentacle (Krakey → user)
"""
from __future__ import annotations

from .sensory import WebChatSensory
from .tentacle import WebChatTentacle


MANIFEST = {
    "name": "web_chat",
    "description": "Dashboard chat channel: user messages arrive as "
                   "user_message stimuli; Krakey replies via "
                   "web_chat_reply. Sensory + tentacle share one "
                   "WebChatHistory (persistence + WS broadcast).",
    "components": [
        {"kind": "sensory", "name": "web_chat",
         "description": "User → Krakey via dashboard WebSocket. "
                        "Pushes user_message stimuli with "
                        "adrenalin=True and channel=web_chat."},
        {"kind": "tentacle", "name": "web_chat_reply",
         "is_internal": False,
         "description": "Krakey → user via dashboard. Outward — the "
                        "text reaches the real human."},
    ],
    "config_schema": [
        {"field": "history_path", "type": "text",
         "default": "workspace/data/web_chat.jsonl",
         "help": "Where the JSONL chat log is persisted. Read by "
                 "Runtime before this plugin loads; editing here "
                 "takes effect on next restart."},
    ],
}


def _shared_history(ctx):
    history = ctx.services.get("web_chat_history")
    if history is None:
        raise RuntimeError(
            "web_chat needs services['web_chat_history']; Runtime "
            "must build WebChatHistory before plugin discovery."
        )
    return history


def build_sensory(ctx):
    """Unified-format factory (Phase 2). User → Krakey via WS."""
    _shared_history(ctx)  # validates the dep exists; sensory uses
                          # the broadcaster wire-up that Runtime sets
                          # up around WebChatHistory, not the history
                          # object itself.
    return WebChatSensory()


def build_tentacle(ctx):
    """Unified-format factory (Phase 2). Krakey → user via WS."""
    history = _shared_history(ctx)
    return WebChatTentacle(history=history)
