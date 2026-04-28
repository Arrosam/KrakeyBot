"""Onboarding wizard — three steps, then write config.yaml.

Walks the user through:
  1. one chat LLM provider (name, base URL, API key, model name) →
     bound to ``self_thinking`` + ``compact`` + ``classifier``
     core purposes via a single ``self_main`` tag.
  2. an optional embedding provider/model (same provider as chat or
     separate). Skippable, but the user is told recall + KB indexing
     need it.
  3. plugin selection. The catalogue is sorted recommended-first,
     dashboard is default-checked and starred — there is no other
     way to inspect Krakey's state without re-running the wizard
     or hand-editing YAML.

Pure stdlib (``input`` / ``print``) — no TUI dependency, works
anywhere Python runs. The runner is fully injectable
(``input_fn`` / ``output_fn`` / ``list_plugins_fn``) so tests can
drive a deterministic happy path without a real TTY.

Idempotent: an existing config.yaml at the target path is backed up
via ``backup_config`` before being overwritten — same backup
mechanism the dashboard uses for its own edits.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from krakey.models.config import (
    Config,
    LLMParams,
    LLMSection,
    Provider,
    TagBinding,
    dump_config,
)
from krakey.models.config_backup import backup_config
from krakey.plugin_system.loader import PluginMetadata
from krakey.plugins.dashboard.services.plugin_catalogue import (
    list_available_plugins,
)


# Plugins the wizard pre-selects + flags as recommended. Currently
# only dashboard — without it there's no in-app way to inspect state
# or change config short of re-running this wizard. New entries here
# also get sorted to the top of the picker.
RECOMMENDED_PLUGINS: tuple[str, ...] = ("dashboard",)


_BANNER = (
    "\n=========================================\n"
    "  Krakey onboarding\n"
    "=========================================\n"
    "Walks you through creating config.yaml.\n"
    "Re-run this any time: krakey onboard\n"
)


InputFn = Callable[[str], str]
OutputFn = Callable[[str], None]
ListPluginsFn = Callable[[], dict[str, PluginMetadata]]


def run_wizard(
    *,
    config_path: Path | str = "config.yaml",
    backup_dir: str = "workspace/backups",
    input_fn: InputFn = input,
    output_fn: OutputFn = print,
    list_plugins_fn: ListPluginsFn = list_available_plugins,
) -> Path:
    """Run the wizard and return the path of the (possibly) written config.

    If the user declines at the final confirm prompt, no file is
    written and the existing config.yaml (if any) is left untouched.
    """
    cfg_path = Path(config_path)

    output_fn(_BANNER)

    chat = _ask_chat_provider(input_fn, output_fn)
    embed = _ask_embedding(input_fn, output_fn, chat)
    plugins = _ask_plugins(input_fn, output_fn, list_plugins_fn())

    cfg = _build_config(chat, embed, plugins)

    if not _confirm_save(input_fn, output_fn, cfg, cfg_path):
        output_fn("aborted; nothing written.")
        return cfg_path

    if cfg_path.exists():
        backup_path = backup_config(cfg_path, backup_dir)
        output_fn(f"backed up existing config -> {backup_path}")
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(dump_config(cfg), encoding="utf-8")
    output_fn(
        f"wrote {cfg_path}. start Krakey with: krakey run"
    )
    return cfg_path


# ---- Step 1: chat provider --------------------------------------

def _ask_chat_provider(
    input_fn: InputFn, output_fn: OutputFn,
) -> tuple[str, Provider, str]:
    output_fn("\n--- Step 1/3: chat LLM provider ---")
    name = _prompt(input_fn, "Provider label (e.g. 'OpenAI')",
                   default="OpenAI")
    base_url = _prompt(
        input_fn, "Base URL (e.g. https://api.openai.com/v1)",
        default="",
    )
    api_key = _prompt(
        input_fn, "API key (blank = leave empty / set later)",
        default="",
    )
    model = _prompt(input_fn, "Model name (e.g. gpt-4o-mini)",
                    default="")
    provider = Provider(
        type="openai_compatible", base_url=base_url, api_key=api_key,
    )
    return name, provider, model


# ---- Step 2: embedding (optional) -------------------------------

def _ask_embedding(
    input_fn: InputFn, output_fn: OutputFn,
    chat: tuple[str, Provider, str],
) -> tuple[str, Provider, str] | None:
    output_fn("\n--- Step 2/3: embedding model (optional) ---")
    output_fn(
        "Embeddings power memory recall and KB indexing. "
        "You can skip and configure later by editing config.yaml.",
    )
    if not _prompt_yes_no(
        input_fn, output_fn, "Configure an embedding model now?",
        default=True,
    ):
        return None
    same = _prompt_yes_no(
        input_fn, output_fn, "Use the same provider as chat?",
        default=False,
    )
    if same:
        emb_name, emb_provider, _ = chat
    else:
        emb_name = _prompt(input_fn, "Embedding provider label",
                            default="SiliconFlow")
        emb_base = _prompt(
            input_fn, "Embedding base URL",
            default="https://api.siliconflow.cn",
        )
        emb_key = _prompt(input_fn, "Embedding API key", default="")
        emb_provider = Provider(
            type="openai_compatible", base_url=emb_base, api_key=emb_key,
        )
    emb_model = _prompt(input_fn, "Embedding model name",
                         default="BAAI/bge-m3")
    return emb_name, emb_provider, emb_model


# ---- Step 3: plugins --------------------------------------------

def _ask_plugins(
    input_fn: InputFn, output_fn: OutputFn,
    available: dict[str, PluginMetadata],
) -> list[str]:
    output_fn("\n--- Step 3/3: plugins ---")
    if not available:
        output_fn("(no plugins discovered)")
        return []
    # Recommended first, then alphabetical. Stable across runs so the
    # user sees the same numbering each time they re-toggle.
    names = sorted(
        available.keys(), key=lambda n: (n not in RECOMMENDED_PLUGINS, n),
    )
    selected: set[str] = {n for n in RECOMMENDED_PLUGINS if n in available}
    while True:
        _print_plugin_list(output_fn, names, available, selected)
        cmd = _prompt(
            input_fn,
            "Toggle a number, or type 'all' / 'none' / 'done'",
            default="done",
        ).strip().lower()
        if cmd == "done":
            break
        if cmd == "all":
            selected = set(names)
            continue
        if cmd == "none":
            selected = set()
            continue
        try:
            idx = int(cmd) - 1
        except ValueError:
            output_fn(f"  ? unknown command: {cmd!r}")
            continue
        if not 0 <= idx < len(names):
            output_fn(f"  ? out of range: {idx + 1}")
            continue
        n = names[idx]
        if n in selected:
            selected.discard(n)
        else:
            selected.add(n)
    return [n for n in names if n in selected]


def _print_plugin_list(
    output_fn: OutputFn, names: list[str],
    available: dict[str, PluginMetadata], selected: set[str],
) -> None:
    output_fn("")
    for i, n in enumerate(names, start=1):
        meta = available[n]
        mark = "[x]" if n in selected else "[ ]"
        star = " *" if n in RECOMMENDED_PLUGINS else "  "
        desc = ""
        if meta.description:
            desc = meta.description.strip().splitlines()[0]
        output_fn(f"  {i:>2}. {mark}{star} {n} - {desc}")


# ---- Build Config -----------------------------------------------

def _build_config(
    chat: tuple[str, Provider, str],
    embed: tuple[str, Provider, str] | None,
    plugin_names: list[str],
) -> Config:
    chat_name, chat_provider, chat_model = chat
    providers = {chat_name: chat_provider}
    tags = {
        "self_main": TagBinding(
            provider=f"{chat_name}/{chat_model}",
            params=LLMParams(),
        ),
    }
    core_purposes = {
        "self_thinking": "self_main",
        "compact": "self_main",
        "classifier": "self_main",
    }
    embedding_tag: str | None = None
    if embed is not None:
        emb_name, emb_provider, emb_model = embed
        if emb_name not in providers:
            providers[emb_name] = emb_provider
        tags["embed"] = TagBinding(
            provider=f"{emb_name}/{emb_model}",
            params=LLMParams(),
        )
        embedding_tag = "embed"
    llm = LLMSection(
        providers=providers, tags=tags,
        core_purposes=core_purposes, embedding=embedding_tag,
    )
    return Config(llm=llm, plugins=(plugin_names or None))


# ---- Preview + confirm ------------------------------------------

def _confirm_save(
    input_fn: InputFn, output_fn: OutputFn,
    cfg: Config, cfg_path: Path,
) -> bool:
    output_fn("\n--- Preview ---")
    output_fn(dump_config(cfg))
    output_fn(f"target: {cfg_path}")
    return _prompt_yes_no(
        input_fn, output_fn, "Write this config?", default=True,
    )


# ---- Prompt helpers ---------------------------------------------

def _prompt(input_fn: InputFn, label: str, *, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    raw = input_fn(f"{label}{suffix}: ").strip()
    return raw or default


def _prompt_yes_no(
    input_fn: InputFn, output_fn: OutputFn, label: str, *, default: bool,
) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    while True:
        raw = input_fn(f"{label} {suffix}: ").strip().lower()
        if not raw:
            return default
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        output_fn(f"  ? please answer y or n (got {raw!r})")
