"""Onboarding wizard — four steps, then write config.yaml.

Walks the user through:
  1. one chat LLM provider (name, base URL, API key, model name) →
     bound to ``self_thinking`` + ``compact`` + ``classifier``
     core purposes via a single ``self_main`` tag.
  2. an optional embedding provider/model (same provider as chat or
     separate). Skippable, but the user is told recall + KB indexing
     need it.
  3. an optional reranker provider/model (same provider as embedding
     or separate). Skippable; the runtime falls back to scripted
     scoring when a reranker isn't available.
  4. plugin selection. The catalogue is sorted recommended-first,
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
from krakey.onboarding import _ui
from krakey.plugin_system.catalogue import list_available_plugins
from krakey.plugin_system.loader import PluginMetadata


# Plugins the wizard pre-selects + flags as recommended. Currently
# only dashboard — without it there's no in-app way to inspect state
# or change config short of re-running this wizard. New entries here
# also get sorted to the top of the picker.
RECOMMENDED_PLUGINS: tuple[str, ...] = ("dashboard",)


_BANNER_FALLBACK = (
    "\n=========================================\n"
    "  Krakey onboarding\n"
    "=========================================\n"
    "Walks you through creating config.yaml.\n"
    "Re-run this any time: krakey onboard\n"
)


def _print_intro(output_fn: OutputFn) -> None:
    """Print the synthwave KRAKEY banner if `output_fn` is the real
    `print` (so the CLI gets the full visual treatment); otherwise
    print a plain text fallback so test fakes that capture output as
    a list of lines stay tidy and grep-able.
    """
    if output_fn is print:
        _ui.enable_vt_on_windows()
        from krakey.cli._banner import print_banner
        print_banner()
        print()
        print(_ui.dim("Walks you through creating config.yaml."))
        print(_ui.dim("Re-run this any time: krakey onboard"))
        print()
    else:
        output_fn(_BANNER_FALLBACK)


def _section(output_fn: OutputFn, title: str) -> None:
    """Print a `--- Step N/4: title ---` header, cyan + bold."""
    output_fn("\n" + _ui.cyan(_ui.bold(f"--- {title} ---")))


InputFn = Callable[[str], str]
OutputFn = Callable[[str], None]
ListPluginsFn = Callable[[], dict[str, PluginMetadata]]
# verify_fn(kind, provider, model) -> (ok, message)
#   kind in {"chat", "embedding", "reranker"}
VerifyFn = Callable[[str, "Provider", str], tuple[bool, str]]
# list_models_fn(provider) -> list[str] | None
#   None on failure (network error, unsupported endpoint, etc.).
ListModelsFn = Callable[["Provider"], "list[str] | None"]


def run_wizard(
    *,
    config_path: Path | str = "config.yaml",
    backup_dir: str = "workspace/backups",
    input_fn: InputFn = input,
    output_fn: OutputFn = print,
    list_plugins_fn: ListPluginsFn = list_available_plugins,
    verify_fn: VerifyFn | None = None,
    list_models_fn: ListModelsFn | None = None,
) -> Path:
    """Run the wizard and return the path of the (possibly) written config.

    If the user declines at the final confirm prompt, no file is
    written and the existing config.yaml (if any) is left untouched.
    """
    cfg_path = Path(config_path)
    verify = verify_fn or _default_verify
    list_models = list_models_fn or _default_list_models

    _print_intro(output_fn)

    chat = _ask_chat_provider(input_fn, output_fn, verify, list_models)
    embed = _ask_embedding(input_fn, output_fn, chat, verify, list_models)
    rerank_cfg = _ask_reranker(
        input_fn, output_fn, chat, embed, verify, list_models,
    )
    plugins = _ask_plugins(input_fn, output_fn, list_plugins_fn())

    # When chat was skipped, the user has NO LLM configured — they
    # depend entirely on the dashboard to fill it in later. Force-add
    # the dashboard plugin if it's available and they unchecked it.
    if chat is None and "dashboard" not in plugins:
        try_available = list_plugins_fn()
        if "dashboard" in try_available:
            output_fn(
                f"  {_ui.yellow('[warn]')} no chat provider configured "
                "AND dashboard is unselected — auto-enabling dashboard "
                "so you have a way to configure providers later."
            )
            plugins = ["dashboard"] + plugins

    cfg = _build_config(chat, embed, rerank_cfg, plugins)

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
    output_fn(
        "\nTip: to right-size the GM node soft-limit for your machine, "
        "run\n  python scripts/bench_gm.py\n"
        "It benchmarks vec_search latency at increasing GM sizes and "
        "prints a recommended `fatigue.gm_node_soft_limit` you can drop "
        "into config.yaml."
    )
    return cfg_path


# ---- Provider-type helper (shared by chat/embedding/reranker) ---

def _ask_provider_type(
    input_fn: InputFn, output_fn: OutputFn, *,
    label: str, default_index: int = 1,
) -> str | None:
    """Ask the user which provider family this {label} talks to.

    Returns one of:
      * ``"openai_compatible"`` — OpenAI, DashScope, llama-server, vllm,
        lmstudio, ollama (OpenAI-compat mode), OneAPI, SiliconFlow, ...
      * ``"anthropic"`` — direct Anthropic API
      * ``None`` — user chose to skip; they'll configure in the dashboard

    ``default_index`` is the 1-based pick when the user just hits Enter.
    """
    output_fn(
        f"\n  Which API does the {label} provider talk?\n"
        "    1. openai_compatible  (OpenAI, DashScope, OneAPI, "
        "SiliconFlow, llama-server, vllm, lmstudio, ollama, ...)\n"
        "    2. anthropic          (Anthropic API directly)\n"
        "    3. skip for now       (configure later in the dashboard)"
    )
    while True:
        raw = _prompt(input_fn, "Choice", default=str(default_index)).strip()
        if raw in ("1", "openai", "openai_compatible"):
            return "openai_compatible"
        if raw in ("2", "anthropic"):
            return "anthropic"
        if raw in ("3", "skip", "s"):
            return None
        output_fn(f"  ? unknown choice: {raw!r}; please answer 1, 2, or 3.")


def _collect_provider_fields(
    input_fn: InputFn, output_fn: OutputFn, *,
    provider_type: str,
    default_label: str, default_base: str, default_model: str,
    list_models: ListModelsFn,
) -> tuple[str, Provider, str]:
    """Prompt for label / base URL / api key, then attempt to enumerate
    models from the provider's `/models` endpoint. On success, present
    a numbered picker (or "type custom"); on failure, fall back to
    plain text entry. Defaults vary by provider_type."""
    name = _prompt(input_fn, "Provider label", default=default_label)
    base_url = _prompt(input_fn, "Base URL", default=default_base)
    api_key = _prompt(
        input_fn, "API key (blank = leave empty / set later)",
        default="",
    )
    provider = Provider(
        type=provider_type, base_url=base_url, api_key=api_key,
    )
    model = _pick_model(
        input_fn, output_fn, provider,
        default_model=default_model, list_models=list_models,
    )
    return name, provider, model


def _pick_model(
    input_fn: InputFn, output_fn: OutputFn, provider: Provider,
    *, default_model: str, list_models: ListModelsFn,
) -> str:
    """Try to list models from the provider; show a picker if listing
    works. Always fall back to plain text entry."""
    models = list_models(provider)
    if not models:
        return _prompt(input_fn, "Model name", default=default_model)
    output_fn(f"\n  Available models ({len(models)}):")
    for i, m in enumerate(models, start=1):
        output_fn(f"    {i:>2}. {m}")
    output_fn("    or type a custom model name.")
    while True:
        raw = _prompt(input_fn, "Model number or name",
                       default=default_model).strip()
        # Numeric pick
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(models):
                return models[idx]
            output_fn(f"  ? out of range: {idx + 1}")
            continue
        except ValueError:
            pass
        # Anything non-empty is treated as a model name (custom).
        if raw:
            return raw
        # Empty + no default → re-prompt
        output_fn("  ? please enter a number or a model name.")


def _skip_hint(output_fn: OutputFn, label: str) -> None:
    output_fn(
        f"  {_ui.dim('[skip]')} {label} not configured. You can fill "
        "it in via the dashboard's LLM tab once Krakey is running, "
        "or by hand-editing config.yaml. Make sure the dashboard "
        "plugin is enabled."
    )


# ---- Step 1: chat provider --------------------------------------

def _ask_chat_provider(
    input_fn: InputFn, output_fn: OutputFn,
    verify: VerifyFn, list_models: ListModelsFn,
) -> tuple[str, Provider, str] | None:
    _section(output_fn, "Step 1/4: chat LLM provider")
    ptype = _ask_provider_type(input_fn, output_fn, label="chat")
    if ptype is None:
        _skip_hint(output_fn, "chat")
        return None
    if ptype == "anthropic":
        defaults = ("Anthropic", "https://api.anthropic.com/v1",
                    "claude-haiku-4-5-20251001")
    else:
        defaults = ("OpenAI", "", "gpt-4o-mini")
    name, provider, model = _collect_provider_fields(
        input_fn, output_fn, provider_type=ptype,
        default_label=defaults[0], default_base=defaults[1],
        default_model=defaults[2], list_models=list_models,
    )
    _report_verify(output_fn, "chat", verify("chat", provider, model))
    return name, provider, model


# ---- Step 2: embedding (optional) -------------------------------

def _ask_embedding(
    input_fn: InputFn, output_fn: OutputFn,
    chat: tuple[str, Provider, str] | None,
    verify: VerifyFn, list_models: ListModelsFn,
) -> tuple[str, Provider, str] | None:
    _section(output_fn, "Step 2/4: embedding model (optional)")
    output_fn(
        "Embeddings power memory recall and KB indexing. "
        "Skip = recall + KB inert (no harm, just no memory benefits).",
    )
    if not _prompt_yes_no(
        input_fn, output_fn, "Configure an embedding model now?",
        default=True,
    ):
        _skip_hint(output_fn, "embedding")
        output_fn(
            f"  {_ui.yellow('[warn]')} Self can't actively recall "
            "topics and sleep won't migrate memories into KBs until "
            "you configure one."
        )
        return None
    if chat is not None:
        same = _prompt_yes_no(
            input_fn, output_fn, "Use the same provider as chat?",
            default=False,
        )
    else:
        same = False
    if same and chat is not None:
        emb_name, emb_provider, _ = chat
    else:
        ptype = _ask_provider_type(
            input_fn, output_fn, label="embedding", default_index=1,
        )
        if ptype is None:
            _skip_hint(output_fn, "embedding")
            return None
        if ptype == "anthropic":
            # Anthropic doesn't ship native embeddings, but allow the
            # user to point at a proxy / aggregator that does.
            defaults = ("Anthropic-proxy", "", "")
        else:
            defaults = ("SiliconFlow",
                         "https://api.siliconflow.cn",
                         "BAAI/bge-m3")
        emb_name, emb_provider, emb_model = _collect_provider_fields(
            input_fn, output_fn, provider_type=ptype,
            default_label=defaults[0], default_base=defaults[1],
            default_model=defaults[2], list_models=list_models,
        )
        _report_verify(output_fn, "embedding",
                        verify("embedding", emb_provider, emb_model))
        return emb_name, emb_provider, emb_model
    # Reused chat provider — model still asked separately.
    emb_model = _pick_model(
        input_fn, output_fn, emb_provider,
        default_model="BAAI/bge-m3", list_models=list_models,
    )
    _report_verify(output_fn, "embedding",
                    verify("embedding", emb_provider, emb_model))
    return emb_name, emb_provider, emb_model


# ---- Step 3: reranker (optional) --------------------------------

def _ask_reranker(
    input_fn: InputFn, output_fn: OutputFn,
    chat: tuple[str, Provider, str] | None,
    embed: tuple[str, Provider, str] | None,
    verify: VerifyFn, list_models: ListModelsFn,
) -> tuple[str, Provider, str] | None:
    """Ask for an optional reranker. Defaults to reusing the embedding
    provider when one is configured (rerankers are commonly served
    alongside embeddings); falls back to chat when there's no
    embedding; falls back to fresh provider config when neither
    earlier step landed.

    Skipping is fine: the runtime degrades gracefully — auto-recall
    falls back to scripted multi-axis scoring, and KB sleep dedup
    falls back to raw cosine ordering.
    """
    _section(output_fn, "Step 3/4: reranker model (optional)")
    output_fn(
        "A reranker improves recall ordering and is also used by "
        "the sleep-time KB dedup pass. Skip to use scripted scoring "
        "+ raw cosine fallback paths instead.",
    )
    if not _prompt_yes_no(
        input_fn, output_fn, "Configure a reranker model now?",
        default=False,
    ):
        return None

    # Default reuse: embedding provider if any, else chat provider, else
    # ask fresh.
    reuse_provider: tuple[str, Provider] | None = None
    reuse_label = ""
    if embed is not None:
        reuse_provider = (embed[0], embed[1])
        reuse_label = "embedding"
    elif chat is not None:
        reuse_provider = (chat[0], chat[1])
        reuse_label = "chat"

    same = False
    if reuse_provider is not None:
        same = _prompt_yes_no(
            input_fn, output_fn,
            f"Use the same provider as {reuse_label}?",
            default=True,
        )

    if same and reuse_provider is not None:
        rer_name, rer_provider = reuse_provider
        rer_model = _pick_model(
            input_fn, output_fn, rer_provider,
            default_model="BAAI/bge-reranker-v2-m3",
            list_models=list_models,
        )
    else:
        ptype = _ask_provider_type(
            input_fn, output_fn, label="reranker", default_index=1,
        )
        if ptype is None:
            _skip_hint(output_fn, "reranker")
            return None
        defaults = ("SiliconFlow",
                     "https://api.siliconflow.cn",
                     "BAAI/bge-reranker-v2-m3")
        rer_name, rer_provider, rer_model = _collect_provider_fields(
            input_fn, output_fn, provider_type=ptype,
            default_label=defaults[0], default_base=defaults[1],
            default_model=defaults[2], list_models=list_models,
        )
    _report_verify(output_fn, "reranker",
                    verify("reranker", rer_provider, rer_model))
    return rer_name, rer_provider, rer_model


# ---- Step 4: plugins --------------------------------------------

def _ask_plugins(
    input_fn: InputFn, output_fn: OutputFn,
    available: dict[str, PluginMetadata],
) -> list[str]:
    _section(output_fn, "Step 4/4: plugins")
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
            # Strong nudge if dashboard ended up unchecked. The
            # dashboard is the only in-app way to inspect Krakey's
            # state and edit config; without it the only recovery
            # path is hand-editing config.yaml or re-running the
            # wizard. Warn and ask to confirm.
            if (
                "dashboard" in available
                and "dashboard" not in selected
            ):
                output_fn(
                    f"\n  {_ui.yellow('[warn]')} dashboard is NOT "
                    "selected. Without it you have no in-app way to "
                    "view runtime state, browse memory, or change "
                    "config — only re-running `krakey onboard` or "
                    "hand-editing config.yaml."
                )
                if not _prompt_yes_no(
                    input_fn, output_fn,
                    "Continue without dashboard?",
                    default=False,
                ):
                    selected.add("dashboard")
                    output_fn("  re-enabled dashboard.")
                    continue
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
        # Colored selection box: green when checked, dim when not.
        mark = _ui.green("[x]") if n in selected else _ui.dim("[ ]")
        # Recommended star: bright yellow + bold, two-space pad for
        # alignment when absent.
        star = _ui.yellow(_ui.bold(" *")) if n in RECOMMENDED_PLUGINS \
            else "  "
        # Plugin name in bold; recommended ones in cyan + bold for
        # extra emphasis.
        if n in RECOMMENDED_PLUGINS:
            name_style = _ui.cyan(_ui.bold(n))
        else:
            name_style = _ui.bold(n)
        output_fn(f"  {i:>2}. {mark}{star} {name_style}")
        # Full description shown under the header line, indented for
        # readability. Multi-line descriptions stay readable;
        # previously we cut to the first line which often hid the
        # actual purpose of a plugin.
        if meta.description:
            for line in meta.description.strip().splitlines():
                output_fn(_ui.dim(f"        {line}"))


# ---- Build Config -----------------------------------------------

def _build_config(
    chat: tuple[str, Provider, str] | None,
    embed: tuple[str, Provider, str] | None,
    rerank_cfg: tuple[str, Provider, str] | None,
    plugin_names: list[str],
) -> Config:
    providers: dict[str, Provider] = {}
    tags: dict[str, TagBinding] = {}
    core_purposes: dict[str, str] = {}
    if chat is not None:
        chat_name, chat_provider, chat_model = chat
        providers[chat_name] = chat_provider
        tags["self_main"] = TagBinding(
            provider=f"{chat_name}/{chat_model}",
            params=LLMParams(),
        )
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
    reranker_tag: str | None = None
    if rerank_cfg is not None:
        rer_name, rer_provider, rer_model = rerank_cfg
        if rer_name not in providers:
            providers[rer_name] = rer_provider
        tags["rerank"] = TagBinding(
            provider=f"{rer_name}/{rer_model}",
            params=LLMParams(),
        )
        reranker_tag = "rerank"
    llm = LLMSection(
        providers=providers, tags=tags,
        core_purposes=core_purposes,
        embedding=embedding_tag, reranker=reranker_tag,
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


# ---- Connectivity verification ----------------------------------

def _report_verify(output_fn: OutputFn, kind: str,
                     result: tuple[bool, str]) -> None:
    """Print verification outcome. Failures are warnings (not aborts)
    so the user can finish the wizard and fix config.yaml manually
    later if a typo only becomes obvious from a real LLM call."""
    ok, msg = result
    if ok:
        output_fn(
            f"  {_ui.green('[check]')} {kind} endpoint reachable ({msg})"
        )
    else:
        output_fn(
            f"  {_ui.yellow('[warn]')}  {kind} verification failed: {msg}"
        )
        output_fn(
            _ui.dim(
                "           continuing — you can fix the config later."
            )
        )


def _default_list_models(provider: Provider) -> list[str] | None:
    """GET `<base>/models` and parse the standard `{"data": [{"id": ...}]}`
    shape that both OpenAI and Anthropic return. Returns the list of
    model ids on success or None on any failure (network error, non-200,
    unparseable body, missing fields). Failure is intentionally silent
    so the wizard cleanly falls back to manual entry — listing is a
    nice-to-have, not a requirement.
    """
    import json
    import socket
    import urllib.error
    import urllib.request

    base = (provider.base_url or "").rstrip("/")
    if not base:
        return None
    url = f"{base}/models"
    headers: dict[str, str] = {}
    if provider.type == "anthropic":
        if provider.api_key:
            headers["x-api-key"] = provider.api_key
        headers["anthropic-version"] = "2023-06-01"
    else:
        if provider.api_key:
            headers["Authorization"] = f"Bearer {provider.api_key}"

    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status != 200:
                return None
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError,
              socket.timeout, OSError, ValueError):
        return None
    except Exception:  # noqa: BLE001
        return None
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        return None
    out = []
    for item in data:
        if isinstance(item, dict) and isinstance(item.get("id"), str):
            out.append(item["id"])
    return out or None


def _default_verify(kind: str, provider: Provider,
                       model: str) -> tuple[bool, str]:
    """Real ping against the provider endpoint. Stdlib only —
    `urllib.request` so onboarding stays sync and dependency-light.

    Sends a minimal real request that exercises base URL + auth +
    model name in one shot. Endpoint + auth header shape vary by
    `provider.type`:

      * openai_compatible:
          chat       → POST /chat/completions  + Authorization: Bearer
          embedding  → POST /embeddings        + Authorization: Bearer
          reranker   → POST /rerank            + Authorization: Bearer
      * anthropic:
          chat       → POST /messages          + x-api-key + anthropic-version
          embedding/reranker: Anthropic API doesn't ship these natively;
          skip the probe and report "skipped".
    """
    import json
    import socket
    import urllib.error
    import urllib.request

    base = (provider.base_url or "").rstrip("/")
    if not base:
        return False, "no base URL configured"

    headers = {"Content-Type": "application/json"}
    if provider.type == "anthropic":
        if kind != "chat":
            return True, "skipped (anthropic native API has no /embeddings or /rerank)"
        url = f"{base}/messages"
        body = {
            "model": model,
            "max_tokens": 1,
            "messages": [{"role": "user", "content": "ping"}],
        }
        if provider.api_key:
            headers["x-api-key"] = provider.api_key
        headers["anthropic-version"] = "2023-06-01"
    else:
        if kind == "chat":
            url = f"{base}/chat/completions"
            body = {
                "model": model,
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 1,
            }
        elif kind == "embedding":
            url = f"{base}/embeddings"
            body = {"model": model, "input": "ping"}
        elif kind == "reranker":
            url = f"{base}/rerank"
            body = {
                "model": model, "query": "ping", "documents": ["test"],
            }
        else:
            return False, f"unknown verify kind {kind!r}"
        if provider.api_key:
            headers["Authorization"] = f"Bearer {provider.api_key}"

    req = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"),
        headers=headers, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            if 200 <= resp.status < 300:
                return True, f"HTTP {resp.status}"
            return False, f"HTTP {resp.status}"
    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code} {e.reason}"
    except (urllib.error.URLError, socket.timeout, OSError) as e:
        return False, f"unreachable: {e}"
    except Exception as e:  # noqa: BLE001
        return False, f"error: {e}"
