"""Console formatter for the heartbeat loop.

Owns ALL print/colour decisions so Runtime stays focused on orchestration.
- white (default): runtime/system events, hibernate, GM stats
- cyan:    Self's inner monologue (decision/thinking/note)
- yellow:  Hypothalamus translation + dispatch
- green:   Every tentacle output (single channel by design — see
           dispatcher.py: the previous internal/chat split was retired
           because a self-declared `Tentacle.is_internal` flag let a
           plugin hide its actions from the operator).
- stderr:  only for warnings (force_sleep, sleep-requested)
"""
from __future__ import annotations

import sys

from src.runtime.console.colors import cyan, green, yellow


class HeartbeatLogger:
    def __init__(self):
        self.heartbeat_id: int = 0

    def set_heartbeat(self, n: int) -> None:
        self.heartbeat_id = n

    # --- runtime / system (white) ---

    def hb(self, msg: str) -> None:
        print(f"[HB #{self.heartbeat_id}] {msg}", flush=True)

    def hb_warn(self, msg: str) -> None:
        print(f"[runtime] {msg}", file=sys.stderr, flush=True)

    def runtime_error(self, msg: str) -> None:
        print(f"[runtime] {msg}", flush=True)

    # --- Self's voice (cyan) ---

    def hb_thought(self, label: str, text: str) -> None:
        print(cyan(f"[HB #{self.heartbeat_id}] {label}: {text.strip()}"),
              flush=True)

    # --- Hypothalamus (yellow) ---

    def hypo(self, msg: str) -> None:
        print(yellow(f"[hypo] {msg}"), flush=True)

    def dispatch(self, msg: str) -> None:
        print(yellow(f"[dispatch] {msg}"), flush=True)

    # --- bot outward chat (green) ---

    def chat(self, sender: str, content: str) -> None:
        print(green(f"[{sender}] {content}"), flush=True)
