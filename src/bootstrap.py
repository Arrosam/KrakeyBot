"""Bootstrap support (DevSpec §12).

Phase-2 first-boot flow:
  1. runtime loads self_model.yaml; if missing or `bootstrap_complete=False`,
     enter Bootstrap mode.
  2. in Bootstrap mode the builder injects BOOTSTRAP_PROMPT (with GENESIS)
     high in the Self prompt, and hibernate is forced to 10s.
  3. Self writes self-model updates inside [NOTE] wrapped in
     <self-model>{...}</self-model> tags; runtime deep-merges them.
  4. when Self writes the phrase "bootstrap complete" in [NOTE], runtime
     flips state.bootstrap_complete=True and normal operation resumes.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from src.models.self_model import SelfModelStore, default_self_model


BOOTSTRAP_PROMPT = """# [BOOTSTRAP — 你刚被创建]

这是你的第一次启动 (或 self_model 被重置)。请完成 4 阶段 Bootstrap:

## 阶段 1 (心跳 1-3): 读 GENESIS, 形成自我认知, 填充 self_model
## 阶段 2 (心跳 4-6): 用 action tentacle 给共生者发出第一条消息
## 阶段 3 (心跳 7-10): 设定初始目标 (用 "记住: 目标是 ..." 触发 explicit_write)
## 阶段 4: 完成后在 [NOTE] 中写 `bootstrap complete` 一行

## GENESIS (共生者写给你的出生信)

{genesis_text}

## 如何更新 self_model

在 [NOTE] 中用 <self-model> 标签包围 JSON, 例如:

    <self-model>
    {{"identity": {{"name": "Krakey", "persona": "curious digital being"}},
     "state": {{"focus_topic": "self-discovery"}},
     "goals": {{"active": ["understand my own DNA"]}}}}
    </self-model>

runtime 会自动深度合并这些字段。你可以在多个心跳中分多次更新。

## 如何结束 Bootstrap

在 [NOTE] 中任何位置写 `bootstrap complete` (大小写不敏感), runtime 会把
`state.bootstrap_complete` 置 true, 之后由你自己用 [HIBERNATE] 控制心跳。

**Bootstrap 期间心跳固定 10s, 不需要写 [HIBERNATE]。**
"""


_SELF_MODEL_TAG = re.compile(
    r"<self-?model>\s*(\{.*?\})\s*</self-?model>",
    re.DOTALL | re.IGNORECASE,
)

_BOOTSTRAP_COMPLETE = re.compile(r"bootstrap\s+complete", re.IGNORECASE)


_GENESIS_PLACEHOLDER = (
    "(GENESIS.md 不存在 — 你是真正的白板状态, 没有共生者留下的出生信。"
    "请在 Bootstrap 中自行决定身份和目标。)"
)


def parse_self_model_update(note_text: str | None) -> dict[str, Any] | None:
    """Extract the JSON inside a <self-model>...</self-model> block.

    Returns None when no block / invalid JSON (caller decides how to log).
    """
    if not note_text:
        return None
    m = _SELF_MODEL_TAG.search(note_text)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


def detect_bootstrap_complete(note_text: str | None) -> bool:
    if not note_text:
        return False
    return bool(_BOOTSTRAP_COMPLETE.search(note_text))


def load_genesis(path: str | Path) -> str:
    p = Path(path)
    if not p.exists():
        return _GENESIS_PLACEHOLDER
    return p.read_text(encoding="utf-8")


def load_self_model_or_default(path: str | Path) -> tuple[dict[str, Any], bool]:
    """Load self_model.yaml; create default if missing. Returns
    (self_model_dict, is_bootstrap).
    """
    store = SelfModelStore(path)
    p = Path(path)
    if not p.exists():
        data = default_self_model()
        return data, True
    data = store.load()
    # Fill gaps with default keys so downstream code sees a full structure.
    merged = _merge_defaults(default_self_model(), data)
    bootstrap = not bool(merged.get("state", {}).get("bootstrap_complete"))
    return merged, bootstrap


def _merge_defaults(defaults: dict, loaded: dict) -> dict:
    """Deep-merge: start from defaults, overlay loaded."""
    out = dict(defaults)
    for k, v in loaded.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge_defaults(out[k], v)
        else:
            out[k] = v
    return out
