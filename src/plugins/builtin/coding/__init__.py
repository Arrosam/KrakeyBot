"""Built-in `coding` plugin — run Python / shell via subprocess or sandbox.

Runner selection (local Subprocess vs sandbox-agent SandboxRunner) is
Runtime policy: the factory calls `deps["build_code_runner"](config)`
to get the right one. If sandbox is requested but unconfigured, that
call raises — loader captures it as the plugin error.
"""
from __future__ import annotations

from src.interfaces.tentacle import Tentacle
from src.tentacles.coding import CodingTentacle


MANIFEST = {
    "name": "coding",
    "description": "Execute a Python or shell command and return exit "
                   "code + stdout / stderr. Routes through the sandbox "
                   "VM when `sandbox: true` (default).",
    "is_internal": True,
    "config_schema": [
        {"field": "enabled",          "type": "bool",   "default": False,
         "help": "Disabled by default; enable only with a configured "
                 "sandbox or on a throwaway machine."},
        {"field": "sandbox",          "type": "bool",   "default": True,
         "help": "When true, exec runs via the sandbox guest agent. "
                 "Set to false only on trusted hosts."},
        {"field": "sandbox_dir",      "type": "text",
         "default": "workspace/sandbox",
         "help": "Working directory hint passed to the runner."},
        {"field": "timeout_seconds",  "type": "number", "default": 30,
         "help": "Subprocess timeout. Exceeding it returns exit=124."},
        {"field": "max_output_chars", "type": "number", "default": 4000,
         "help": "stdout / stderr truncated past this many chars."},
    ],
}


def create_tentacle(config: dict, deps: dict) -> Tentacle:
    build_runner = deps.get("build_code_runner")
    if build_runner is None:
        raise RuntimeError(
            "coding plugin needs deps['build_code_runner'] "
            "(Runtime._build_code_runner callable)."
        )
    runner = build_runner(config)
    return CodingTentacle(
        runner=runner,
        sandbox_dir=str(config.get("sandbox_dir", "workspace/sandbox")),
        timeout_seconds=int(config.get("timeout_seconds", 30)),
        max_output_chars=int(config.get("max_output_chars", 4000)),
    )
