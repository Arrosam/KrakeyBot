"""Built-in Telegram project — sensory + outbound reply tentacle.

Multi-component project: the sensory polls incoming messages, the
tentacle sends replies, and both share one HttpTelegramClient instance.
This is exactly the kind of plugin that needed the `create_plugins`
factory — a factory-per-component would build two clients and lose
rate-limit / connection coordination.

Layout within this package:
  - client.py   — HttpTelegramClient + TelegramClient Protocol
  - sensory.py  — TelegramSensory (inbound polling)
  - tentacle.py — TelegramReplyTentacle (outbound send)
"""
from __future__ import annotations

from .client import HttpTelegramClient
from .sensory import TelegramSensory
from .tentacle import TelegramReplyTentacle


MANIFEST = {
    "name": "telegram",
    "description": "Telegram bidirectional channel: incoming messages "
                   "arrive as user_message stimuli; Krakey replies via "
                   "telegram_reply. Sensory + tentacle share one HTTP "
                   "client.",
    "components": [
        {"kind": "sensory",  "name": "telegram",
         "description": "Polls Telegram getUpdates; pushes user_message "
                        "stimuli into the buffer."},
        {"kind": "tentacle", "name": "telegram_reply",
         "is_internal": False,
         "description": "Sends a Telegram message. Outward — the text "
                        "reaches the real human."},
    ],
    "config_schema": [
        {"field": "bot_token", "type": "password", "default": "",
         "help": "BotFather token. Use ${ENV_VAR} to pull from the "
                 "environment at load time."},
        {"field": "allowed_chat_ids", "type": "text", "default": "",
         "help": "Comma-separated chat IDs the sensory will accept. "
                 "Empty = allow any."},
        {"field": "default_chat_id", "type": "text", "default": "",
         "help": "Chat ID the reply tentacle sends to when Hypothalamus "
                 "did not specify one."},
    ],
}


def _parse_allowed(raw) -> set[int] | None:
    if raw is None:
        return None
    if isinstance(raw, list):
        return {int(x) for x in raw}
    s = str(raw).strip()
    if not s:
        return None
    return {int(x.strip()) for x in s.split(",") if x.strip()}


def _shared_client(ctx) -> HttpTelegramClient:
    """Build (or fetch from plugin_cache) the shared HTTP client.
    Both build_sensory and build_tentacle call this; first call
    constructs, subsequent calls hit the cache so the sensory and
    the tentacle both talk through the same connection."""
    if "client" not in ctx.plugin_cache:
        token = str(ctx.config.get("bot_token") or "").strip()
        if not token:
            raise RuntimeError(
                "telegram plugin requires bot_token (use env var via "
                "${TELEGRAM_BOT_TOKEN} in config.yaml)."
            )
        ctx.plugin_cache["client"] = HttpTelegramClient(token=token)
    return ctx.plugin_cache["client"]


def build_sensory(ctx):
    """Unified-format factory (Phase 2). Inbound polling channel."""
    client = _shared_client(ctx)
    allowed = _parse_allowed(ctx.config.get("allowed_chat_ids"))
    return TelegramSensory(client=client, allowed_chat_ids=allowed)


def build_tentacle(ctx):
    """Unified-format factory (Phase 2). Outbound reply channel."""
    client = _shared_client(ctx)
    default_chat_raw = str(ctx.config.get("default_chat_id") or "").strip()
    default_chat = int(default_chat_raw) if default_chat_raw else None
    return TelegramReplyTentacle(client=client, default_chat_id=default_chat)
