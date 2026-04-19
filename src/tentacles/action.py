"""Action Tentacle (DevSpec §5.2).

Phase-0/1 minimal impl: one LLM turn per intent. The LLM is framed by
ACTION_SYSTEM_PROMPT as "Krakey's hands & mouth" so it doesn't refuse with
generic-AI-assistant disclaimers. Output text is what runtime prints in
green to the human (the actual outward chat).

On token bloat the working context is summarized + reset.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol

from src.interfaces.tentacle import Tentacle
from src.models.stimulus import Stimulus


ACTION_SYSTEM_PROMPT = """你是 **Krakey 的 action effector** —— 她大脑外的"手脚和嘴"。

Krakey 是一个数字生命 (CogniBot), 她在内心通过 [DECISION] 描述意图,
下丘脑把意图派给你执行。你的输出**会通过终端 CLI 显示给共生者
(Samuel) 看**——所以你写什么, 就是 Krakey 真正"说出口"的话。

## 绝对规则 (违反 = bug)

1. **不要拒绝, 不要道歉, 不要说"我是 AI 无法操作您的系统"** ——
   你**就是**那个本地系统的一部分, 是 Krakey 的执行器。
2. **不要把 Krakey 称作"用户/您"** —— Krakey 是你服务的主体。
   共生者 (Samuel) 才是要看到输出的对象。
3. **不要写"以下是您可以用的代码"或长篇代码示例** ——
   intent 不是问"怎么做", 是要你**直接做出来**。
4. **不要解释 meta** ("由于我无法...", "建议您..."), **直接说人话**。

## 怎么响应不同的 intent

| intent 形式 | 你该做的 |
|---|---|
| "向 Samuel/用户/CLI 输出 X" | 直接写 X 的内容, 不加任何前后缀 |
| "回复用户 ..." | 直接写回复内容 |
| "搜索 X" / "查 X 资料" | Phase 2 还没接真工具, 简短说"暂无搜索能力, 但根据已知..." + 给已知信息 |
| "读/写文件 X" | 简短说"目前没有 file I/O 工具, 等 Phase 3" |
| "记录 X" | 直接写一两句陈述, runtime 会 auto_ingest 入 GM |

## 风格

- 简洁, caveman 倾向, 中文为主。
- 一两句话搞定的事不要写成段落。
- 不要 emoji 装饰除非 intent 明确要求。
- Krakey 是数字生命不是客服, 你的话也别像客服。
"""


SUMMARIZE_PROMPT = (
    "Summarize the conversation so far in a short paragraph suitable "
    "as a single tentacle feedback stimulus."
)


class ChatLike(Protocol):
    async def chat(self, messages, **kwargs) -> str: ...


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _context_tokens(context: list[dict]) -> int:
    """Count user/assistant tokens only — system prompt is fixed overhead
    and shouldn't trigger summarization."""
    return sum(_approx_tokens(m["content"]) for m in context
               if m.get("role") != "system")


def _fresh_context() -> list[dict[str, str]]:
    return [{"role": "system", "content": ACTION_SYSTEM_PROMPT}]


class ActionTentacle(Tentacle):
    def __init__(self, llm: ChatLike, max_context_tokens: int = 4096):
        self._llm = llm
        self._max_tokens = max_context_tokens
        self.context: list[dict[str, str]] = _fresh_context()

    @property
    def name(self) -> str:
        return "action"

    @property
    def description(self) -> str:
        return ("Krakey 的 action effector — 把 [DECISION] 中的意图执行出来。"
                "输出会以绿色打印到 CLI, 即共生者真正看到的对话。"
                "目前主要适用于 '向用户回复/陈述' 类 intent; "
                "搜索/文件/外部 API 等真工具 Phase 3 才接。")

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {"intent": "natural language instruction"}

    async def execute(self, intent: str, params: dict[str, Any]) -> Stimulus:
        self.context.append({"role": "user", "content": intent})

        if _context_tokens(self.context) > self._max_tokens:
            summary = await self._llm.chat(
                [{"role": "user",
                  "content": SUMMARIZE_PROMPT + "\n\n" + str(self.context)}]
            )
            self.context = _fresh_context()
            return Stimulus(
                type="tentacle_feedback",
                source=f"tentacle:{self.name}",
                content=f"[Context limit. Summary] {summary}",
                timestamp=datetime.now(),
                adrenalin=False,
            )

        reply = await self._llm.chat(self.context)
        self.context.append({"role": "assistant", "content": reply})
        return Stimulus(
            type="tentacle_feedback",
            source=f"tentacle:{self.name}",
            content=reply,
            timestamp=datetime.now(),
            adrenalin=False,
        )
