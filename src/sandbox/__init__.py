"""Sandbox — host-side plumbing for routing non-idempotent tentacles
(coding, GUI, file I/O, browser, CLI) through a guest agent running
inside a VM.

Phase S1 scope: `exec(cmd, timeout)` RPC only, used by the coding
tentacle when `tentacle.coding.sandbox` is true.

The guest agent (sandbox/agent.py) must already be running inside the
user's VM before Krakey starts, with network visible on the host-only
subnet configured under `sandbox.agent.url`.
"""
