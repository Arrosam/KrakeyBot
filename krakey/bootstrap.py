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

import copy
import json
import re
from pathlib import Path
from typing import Any

from krakey.models.self_model import SelfModelStore, default_self_model


BOOTSTRAP_PROMPT = """# [BOOTSTRAP — 你刚被创建]

这是你的第一次启动 (或 self_model 被重置)。请完成 4 阶段 Bootstrap:

## 阶段 1 (心跳 1-3): 读 GENESIS, 形成自我认知, 填充 self_model
## 阶段 2 (心跳 4-6): 用 action tool 给共生者发出第一条消息
## 阶段 3 (心跳 7-10): 设定初始目标 (用 "记住: 目标是 ..." 触发 explicit_write)
## 阶段 4: 完成后在 [NOTE] 中写 `bootstrap complete` 一行

## GENESIS (共生者写给你的出生信)

{genesis_text}

## 如何更新 self_model

self_model 现在只保留两块**真正不变的核心**: identity (你叫什么、你是什么)
和 state.bootstrap_complete (Bootstrap 是否结束的开关)。

当下专注的事 / 目标 / 关系 / 情绪状态等**不进 self_model** —— 它们的真相
在 Graph Memory 里 (FOCUS / TARGET 节点 + 边)。Bootstrap 期间你只需要用
<self-model> 标签写好你的 identity, 例如:

    <self-model>
    {{"identity": {{"name": "Krakey", "persona": "curious digital being"}}}}
    </self-model>

runtime 会自动深度合并。Bootstrap 之外, identity 通常一辈子不再变。

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

    On load, any keys present in the YAML but **not** in the current
    ``default_self_model()`` schema are silently dropped — this is the
    one-shot migration for the 2026-04-25 self-model slim refactor.
    Legacy fields like ``statistics.*``, ``relationships.users``,
    ``state.mood_baseline``, ``state.is_sleeping``, ``state.focus_topic``,
    ``state.energy_level``, ``goals.active``, and ``goals.completed``
    were never read in steady state; keeping them around just bloated
    every Self prompt for no behavioral benefit. If anything was
    dropped, we save the cleaned version back to disk so next boot
    is fast and the YAML modifiers what's actually in use.
    """
    import logging

    store = SelfModelStore(path)
    p = Path(path)
    if not p.exists():
        data = default_self_model()
        return data, True
    data = store.load()
    merged = _merge_defaults(default_self_model(), data)
    bootstrap = not bool(merged.get("state", {}).get("bootstrap_complete"))
    if merged != data:
        # Persist the migration so the file is self-consistent next run.
        # Logging at INFO so the breadcrumb is visible without becoming
        # noise on every subsequent boot (after migration the dicts are
        # equal and we skip the write).
        dropped = _diff_keys(data, merged)
        logging.getLogger(__name__).info(
            "self_model migration: dropped legacy keys %s; rewriting %s",
            dropped, path,
        )
        store.save(merged)
    return merged, bootstrap


def _merge_defaults(defaults: dict, loaded: dict) -> dict:
    """Left-bounded deep-merge.

    Only keys present in ``defaults`` survive. Loaded values overlay
    defaults where the key matches; loaded keys not in defaults are
    silently dropped. This is what makes ``load_self_model_or_default``
    auto-migrate: the slim ``default_self_model()`` schema acts as
    the authoritative key set and old YAMLs get pruned on first read.
    """
    out: dict[str, Any] = {}
    for k, default_v in defaults.items():
        if k not in loaded:
            out[k] = copy.deepcopy(default_v)
            continue
        loaded_v = loaded[k]
        if isinstance(default_v, dict) and isinstance(loaded_v, dict):
            out[k] = _merge_defaults(default_v, loaded_v)
        else:
            out[k] = loaded_v
    return out


def _diff_keys(loaded: dict, merged: dict, prefix: str = "") -> list[str]:
    """Return dotted-paths for every key that appears in ``loaded`` but
    not in ``merged``. Used only for the one-time migration log line.
    """
    out: list[str] = []
    for k, v in loaded.items():
        path = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if k not in merged:
            out.append(path)
        elif isinstance(v, dict) and isinstance(merged.get(k), dict):
            out.extend(_diff_keys(v, merged[k], prefix=path))
    return out
