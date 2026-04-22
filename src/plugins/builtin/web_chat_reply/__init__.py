"""Built-in `web_chat_reply` plugin — Krakey → dashboard chat outbound."""
from __future__ import annotations

from src.interfaces.tentacle import Tentacle
from src.tentacles.web_chat_reply import WebChatTentacle


MANIFEST = {
    "name": "web_chat_reply",
    "description": "Send a message into the web chat (dashboard). "
                   "Outward: the text reaches the human directly.",
    "is_internal": False,
    "config_schema": [
        {"field": "enabled", "type": "bool", "default": True},
        # history_path is owned by Runtime (it constructs WebChatHistory
        # before plugins load), not the plugin factory. Kept off the
        # schema to avoid implying the field is editable here.
    ],
}


def create_tentacle(config: dict, deps: dict) -> Tentacle:
    history = deps.get("web_chat_history")
    if history is None:
        raise RuntimeError(
            "web_chat_reply needs web_chat_history in deps; runtime did "
            "not build WebChatHistory before plugin discovery."
        )
    return WebChatTentacle(history=history)
