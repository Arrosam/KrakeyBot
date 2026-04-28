"""Built-in Telegram plugin — sensory + outbound reply tool.

Multi-component plugin: the sensory polls incoming messages, the
tool sends replies, and both share one HttpTelegramClient instance.
``build_sensory`` and ``build_tool`` cooperate via ``ctx.plugin_cache``
so the first factory call constructs the client and the second reuses
it — a factory-per-component would otherwise build two clients and lose
rate-limit / connection coordination.

Layout within this package:
  - client.py   — HttpTelegramClient + TelegramClient Protocol
  - sensory.py  — TelegramSensory (inbound polling)
  - tool.py — TelegramReplyTool (outbound send)
"""
from __future__ import annotations

from .client import HttpTelegramClient
from .sensory import TelegramSensory
from .tool import TelegramReplyTool


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
    Both build_sensory and build_tool call this; first call
    constructs, subsequent calls hit the cache so the sensory and
    the tool both talk through the same connection."""
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


def build_tool(ctx):
    """Unified-format factory (Phase 2). Outbound reply channel."""
    client = _shared_client(ctx)
    default_chat_raw = str(ctx.config.get("default_chat_id") or "").strip()
    default_chat = int(default_chat_raw) if default_chat_raw else None
    return TelegramReplyTool(client=client, default_chat_id=default_chat)
