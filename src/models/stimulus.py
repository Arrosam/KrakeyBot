"""Stimulus dataclass (DevSpec §6.1)."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class Stimulus:
    type: str                # user_message | tentacle_feedback | batch_complete | system_event
    source: str              # sensory:cli | tentacle:web_chat_reply | sensory:batch_tracker
    content: str
    timestamp: datetime
    adrenalin: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
