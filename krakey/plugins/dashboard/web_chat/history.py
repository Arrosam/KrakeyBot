"""Phase 3.F.3: WebChatHistory — persistent + broadcast-aware chat store.

Single source of truth for the web chat:
  - persists to JSONL on disk (so messages survive restarts)
  - holds an in-memory cache for fast initial-load on WS connect
  - broadcasts every appended message to subscribed callbacks (the WS
    endpoints push to connected browsers; the WebChatTool and
    WebChatChannel both use the same store)
"""
from __future__ import annotations

import inspect
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


class WebChatHistory:
    def __init__(self, jsonl_path: str | Path):
        self.path = Path(jsonl_path)
        self._cache: list[dict[str, Any]] = self._load_from_disk()
        self._subscribers: list[Callable[[dict[str, Any]], Any]] = []

    def _load_from_disk(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        out: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out

    def all_messages(self) -> list[dict[str, Any]]:
        return list(self._cache)

    async def append(self, sender: str, content: str,
                       attachments: list[dict[str, Any]] | None = None,
                       message_id: str | None = None,
                       status: str | None = None,
                       ) -> dict[str, Any]:
        """Persist + broadcast a message. Returns the persisted record.

        `attachments`: optional list of {name, url, type, size} dicts as
        returned by /api/chat/upload.
        `message_id`: optional stable ID for this message (user messages
        only; krakey replies are never keyed).
        `status`: optional delivery status string ("delivered", "failed",
        "read"); omitted when not provided.
        """
        msg: dict[str, Any] = {
            "sender": sender,
            "content": content,
            "ts": datetime.now().isoformat(),
        }
        if message_id is not None:
            msg["id"] = message_id
        if status is not None:
            msg["status"] = status
        if attachments:
            msg["attachments"] = attachments
        self._cache.append(msg)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(msg, ensure_ascii=False) + "\n")
        await self._broadcast(msg)
        return msg

    def _rewrite_from_cache(self) -> None:
        """Truncate the JSONL file and rewrite every cached record.

        Used by update_status and delete to keep the on-disk file in
        sync after in-place mutations.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as f:
            for record in self._cache:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    async def update_status(self, message_id: str, status: str) -> "dict[str, Any] | None":
        """Find the cached record with ``id == message_id``, set its
        status, rewrite the JSONL file, and return the updated record.
        Returns None if no matching record is found.
        """
        for record in self._cache:
            if record.get("id") == message_id:
                record["status"] = status
                self._rewrite_from_cache()
                return record
        return None

    async def delete(self, message_id: str) -> bool:
        """Remove the cached record with ``id == message_id`` (if any),
        rewrite the JSONL file, and return True if removed else False.
        """
        for i, record in enumerate(self._cache):
            if record.get("id") == message_id:
                del self._cache[i]
                self._rewrite_from_cache()
                return True
        return False

    async def _broadcast(self, msg: dict[str, Any]) -> None:
        for cb in list(self._subscribers):
            try:
                if inspect.iscoroutinefunction(cb):
                    await cb(msg)
                else:
                    cb(msg)
            except Exception:  # noqa: BLE001
                # WebSocket dropped, broken subscriber — just skip
                pass

    def subscribe(self, cb: Callable[[dict[str, Any]], Any]) -> None:
        self._subscribers.append(cb)

    def unsubscribe(self, cb: Callable[[dict[str, Any]], Any]) -> None:
        try:
            self._subscribers.remove(cb)
        except ValueError:
            pass
