"""web_chat package — data layer + WS route.

`WebChatHistory` is the single source of truth for the web chat
stream (JSONL persistence + broadcast subscription). The WS route
lives alongside it because both touch the same object and nothing
else in the codebase does.

Re-exporting `WebChatHistory` here keeps imports short: both the
dashboard wiring and the `web_chat` plugin read
`from src.dashboard.web_chat import WebChatHistory`.
"""
from src.dashboard.web_chat.history import WebChatHistory
from src.dashboard.web_chat.service import RuntimeWebChatService

__all__ = ["WebChatHistory", "RuntimeWebChatService"]
