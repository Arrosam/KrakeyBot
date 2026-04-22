"""Built-in Telegram project — sensory + outbound reply tentacle.

Multi-component project: the sensory polls incoming messages, the
tentacle sends replies, and both share one HttpTelegramClient instance.
This is exactly the kind of plugin that needed the `create_plugins`
factory — a factory-per-component would build two clients and lose
rate-limit / connection coordination.
"""
from __future__ import annotations

from src.sensories.telegram import HttpTelegramClient, TelegramSensory
from src.tentacles.telegram_reply import TelegramReplyTentacle


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
        {"field": "enabled", "type": "bool", "default": False,
         "help": "Master switch for both the sensory and the reply "
                 "tentacle."},
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


def create_plugins(config: dict, deps: dict) -> dict:
    token = str(config.get("bot_token") or "").strip()
    if not token:
        raise RuntimeError(
            "telegram plugin requires bot_token (use env var via "
            "${TELEGRAM_BOT_TOKEN} in config.yaml)."
        )
    client = HttpTelegramClient(token=token)
    allowed = _parse_allowed(config.get("allowed_chat_ids"))
    default_chat_raw = str(config.get("default_chat_id") or "").strip()
    default_chat = int(default_chat_raw) if default_chat_raw else None

    sensory = TelegramSensory(client=client, allowed_chat_ids=allowed)
    tentacle = TelegramReplyTentacle(client=client,
                                        default_chat_id=default_chat)
    return {"tentacles": [tentacle], "sensories": [sensory]}
