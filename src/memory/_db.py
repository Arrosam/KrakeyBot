"""Shared persistence helpers for GM + KB.

Extracted so KnowledgeBase doesn't reach into graph_memory's internals.
Pure helpers — no schema knowledge of either GM or KB tables.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import aiosqlite
import sqlite_vec


SCHEMA_PATH = Path(__file__).parent / "schemas.sql"


# ---------------- connection ----------------


async def open_db_with_vec(path: str | Path) -> aiosqlite.Connection:
    """Open an aiosqlite connection, load sqlite-vec, enable FK + Row factory."""
    db = await aiosqlite.connect(str(path))
    db.row_factory = aiosqlite.Row
    await db.enable_load_extension(True)
    await db.load_extension(sqlite_vec.loadable_path())
    await db.enable_load_extension(False)
    await db.execute("PRAGMA foreign_keys = ON")
    return db


async def apply_schema(db: aiosqlite.Connection,
                          schema_text: str | None = None) -> None:
    """Run the bundled schemas.sql (or a provided SQL string) and commit.

    Also runs in-place ALTER TABLE for columns added after a DB was first
    created — SQLite has no `ADD COLUMN IF NOT EXISTS`, so we PRAGMA-check
    each evolving column and add it on demand. Keep this list small and
    only for additive changes.
    """
    if schema_text is None:
        schema_text = SCHEMA_PATH.read_text(encoding="utf-8")
    await db.executescript(schema_text)
    await _ensure_columns(db, "kb_registry", {
        "is_archived": "INTEGER DEFAULT 0",
        "index_embedding": "BLOB",
    })
    await db.commit()


async def _ensure_columns(db: aiosqlite.Connection, table: str,
                            columns: dict[str, str]) -> None:
    async with db.execute(f"PRAGMA table_info({table})") as cur:
        existing = {row[1] for row in await cur.fetchall()}
    if not existing:
        # Table doesn't exist in this DB (e.g. KB-only file) — nothing to do
        return
    for col, ddl in columns.items():
        if col not in existing:
            await db.execute(f"ALTER TABLE {table} ADD COLUMN {col} {ddl}")


# ---------------- vector helpers ----------------


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity. 0.0 when either vector is zero (no NaN)."""
    if len(a) != len(b):
        raise ValueError(f"vector length mismatch: {len(a)} vs {len(b)}")
    dot = na = nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def encode_embedding(vec: list[float] | None) -> bytes | None:
    """JSON-encode a vector for BLOB storage."""
    if vec is None:
        return None
    return json.dumps(list(vec)).encode("utf-8")


def decode_embedding(blob: bytes | None) -> list[float] | None:
    if blob is None:
        return None
    return json.loads(blob.decode("utf-8"))


# ---------------- FTS5 query construction ----------------


_FTS_TOKEN = re.compile(r"\w+", re.UNICODE)


def build_fts_query(text: str | None) -> str | None:
    """Quote each token + OR them together. Strips MATCH operators so
    user input never trips FTS5 syntax errors."""
    if not text:
        return None
    tokens = _FTS_TOKEN.findall(text)
    if not tokens:
        return None
    return " OR ".join(f'"{t}"' for t in tokens)
