"""Layer-0 DNA instructions — loaded once from sibling ``dna.txt``.

Fixed system prompt. Not user-configurable at runtime.

DNA = mechanics only. Input/output format, perception, action, memory,
sleep, inner voice. No identity. No behavioral norms. Identity lives
in GENESIS (Bootstrap once) and persists via self_model.

The text itself lives in ``krakey/prompt/dna.txt`` so prose edits get
markdown-friendly editor support + clean line diffs (no escaping +
no quote-string indentation noise). This module just reads the file
once at import time and exposes ``DNA`` as a module-level constant
— exactly the same import surface as before:

    from krakey.prompt.dna import DNA
"""
from __future__ import annotations

from pathlib import Path


DNA: str = (Path(__file__).parent / "dna.txt").read_text(encoding="utf-8")
