"""Console formatter for the heartbeat loop.

Owns ALL print/colour decisions so Runtime stays focused on orchestration.
- white (default): runtime/system events, hibernate, GM stats
- cyan:    Self's inner monologue (decision/thinking/note)
- yellow:  Hypothalamus translation + dispatch
- magenta: internal tentacle returns (memory_recall etc. — not chat)
- green:   Bot's outward chat replies (the only thing the human reads as
           Krakey's voice)
- stderr:  only for warnings (force_sleep, sleep-requested)
"""
from __future__ import annotations

import sys

from src.runtime.colors import cyan, green, magenta, yellow


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

    def hypo_warn(self, msg: str) -> None:
        print(yellow(f"[hypo] {msg}"), file=sys.stderr, flush=True)

    def dispatch(self, msg: str) -> None:
        print(yellow(f"[dispatch] {msg}"), flush=True)

    # --- internal tentacle returns (magenta) ---

    def internal(self, sender: str, content: str) -> None:
        """For tentacles whose output is for Self only, not user-facing
        chat (e.g. memory_recall results)."""
        print(magenta(f"[{sender}] {content}"), flush=True)

    # --- bot outward chat (green) ---

    def chat(self, sender: str, content: str) -> None:
        print(green(f"[{sender}] {content}"), flush=True)
