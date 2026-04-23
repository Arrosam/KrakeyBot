"""ConfigService — Protocol for settings read / write / restart."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol


class ConfigService(Protocol):
    """Encapsulate config.yaml I/O + backup + restart hook.

    Keeps yaml parsing, backup policy, and the restart callback behind
    one seam. The route just translates HTTP verbs to method calls and
    maps exceptions to HTTPException.
    """

    @property
    def path(self) -> Path | None: ...

    def read(self) -> tuple[str, Any]:
        """Returns (raw_yaml_string, parsed_python_object_or_None)."""
        ...

    def write(
        self, *, raw: str | None, parsed: Any, backup_dir: str,
    ) -> Path | None:
        """Serialize `parsed` (preferred) or validate + write `raw`.

        Returns the backup file path (or None if backup was skipped).
        Raises ValueError on invalid input.
        """
        ...

    def restart(self) -> None:
        """Trigger an in-process restart, or raise RuntimeError if not
        wired (test fixtures / headless mode)."""
        ...
