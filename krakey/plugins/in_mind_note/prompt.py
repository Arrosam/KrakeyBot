"""Prompt-side rendering for the in_mind Reflect.

Two pieces:

  * ``IN_MIND_INSTRUCTIONS_LAYER`` — constant string injected as a
    standalone prompt layer (between [CAPABILITIES] and [STIMULUS]).
    Cache-friendly: only present / absent depending on whether the
    in_mind Reflect is registered, never changing content within a
    run. Its job is to teach Self **when** to call ``update_in_mind``.
  * ``render_virtual_round(state)`` — formats the per-beat virtual
    round prepended to [HISTORY] when at least one in_mind field is
    populated.
"""
from __future__ import annotations

from krakey.plugins.in_mind_note.state import InMindState


IN_MIND_INSTRUCTIONS_LAYER = """# [IN MIND — 操作约束]
你的"心智状态" (Thoughts / Mood / Focus) 是别的系统读取你"现在心里在
想什么"的唯一来源, 持续显示在 [HISTORY] 头部 "Heartbeat #now (in mind)"
块里。

每当下列任一情况发生, 你必须立刻调用 update_in_mind tool 更新对应
字段:

- 你的思考焦点切换 (新话题 / 新问题 / 新线索)
- 情绪明显变化
- 专注的事改变

只更新变化的字段。三个参数都是可选的:

  thoughts: 当前心头最重要的事 (一句话即可)
  mood:     当前情绪 + 简短原因
  focus:    正在专注的具体事

不传一个字段 = 该字段不动; 传空字符串 = 显式清空该字段。

不更新 = 别的系统看到的是过期信息, 会做错决策。这是常驻指令, 每跳都
适用, 不需要等被提醒。"""


def render_virtual_round(state: InMindState) -> str | None:
    """Return the multi-line block to prepend at the head of
    [HISTORY], or None if every field is empty (no point inserting
    a round of nothing — wastes prompt tokens + adds visual noise).
    """
    if state.is_empty():
        return None
    lines = ["--- Heartbeat #now (in mind) ---"]
    # Render only non-empty fields so Self isn't told "Mood: " on a
    # line. Empty = "I never set this", not "I am literally feeling
    # nothing".
    if state.thoughts:
        lines.append(f"Thoughts: {state.thoughts}")
    if state.mood:
        lines.append(f"Mood: {state.mood}")
    if state.focus:
        lines.append(f"Focus: {state.focus}")
    return "\n".join(lines)
