"""Phase 3.F.6: rolling backups of config.yaml.

Run on every Krakey startup so a bad save (via the dashboard Settings
page) can always be reverted from `workspace/backups/`.
"""
from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path


BACKUP_FILENAME_PREFIX = "config-"
BACKUP_FILENAME_SUFFIX = ".yaml"


def backup_config(src: str | Path, backup_dir: str | Path,
                    *, keep_last: int = 10) -> Path | None:
    """Copy `src` (config.yaml) into `backup_dir` with a timestamped name.

    Returns the new backup path, or None when the source is missing.
    Old backups beyond `keep_last` are deleted (oldest first).
    """
    src_path = Path(src)
    if not src_path.exists():
        return None
    dir_path = Path(backup_dir)
    dir_path.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    dest = dir_path / f"{BACKUP_FILENAME_PREFIX}{ts}{BACKUP_FILENAME_SUFFIX}"
    shutil.copy2(src_path, dest)
    _trim_old(dir_path, keep_last)
    return dest


def list_backups(backup_dir: str | Path) -> list[Path]:
    """Existing backups, newest first."""
    d = Path(backup_dir)
    if not d.exists():
        return []
    items = [p for p in d.iterdir()
             if p.is_file() and p.name.startswith(BACKUP_FILENAME_PREFIX)
             and p.name.endswith(BACKUP_FILENAME_SUFFIX)]
    items.sort(key=lambda p: p.name, reverse=True)
    return items


def _trim_old(dir_path: Path, keep_last: int) -> None:
    backups = list_backups(dir_path)
    for old in backups[keep_last:]:
        try:
            old.unlink()
        except OSError:
            pass
