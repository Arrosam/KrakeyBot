"""``heartbeat`` Engine — per-beat orchestration + main loop driver.

The slot's catalog of impls lives in ``meta.yaml`` next to this file.
The HeartbeatEngine Protocol lives at
``krakey.interfaces.engines.heartbeat``.
"""
from krakey.engines.heartbeat.default import DefaultHeartbeatEngine

__all__ = ["DefaultHeartbeatEngine"]
