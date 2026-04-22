"""Built-in `web_chat_reply` plugin — Krakey → dashboard chat outbound.

Single-component project. The shared WebChatHistory is built by the
Runtime before plugin discovery (the dashboard WebSocket subscribes to
it too, so it must exist earlier than this plugin loads) and handed
in via deps["web_chat_history"].
"""
from __future__ import annotations

from src.interfaces.tentacle import Tentacle
from src.tentacles.web_chat_reply import WebChatTentacle


MANIFEST = {
    "name": "web_chat_reply",
    "description": "Send a message into the dashboard web chat. "
                   "Outward — the text reaches the real human.",
    "is_internal": False,
    "config_schema": [
        {"field": "enabled", "type": "bool", "default": True},
        {"field": "history_path", "type": "text",
         "default": "workspace/data/web_chat.jsonl",
         "help": "Where the JSONL chat log is persisted. Read by "
                 "Runtime before this plugin loads; editing here "
                 "takes effect on next restart."},
    ],
}


def create_tentacle(config: dict, deps: dict) -> Tentacle:
    history = deps.get("web_chat_history")
    if history is None:
        raise RuntimeError(
            "web_chat_reply needs deps['web_chat_history']; Runtime "
            "must build WebChatHistory before plugin discovery."
        )
    return WebChatTentacle(history=history)
