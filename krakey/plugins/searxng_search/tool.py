"""``searxng_search`` Tool — query a local SearXNG aggregator instance.

Pure scripted dispatch. No inner LLM. Every failure mode (bad params,
unreachable instance, non-JSON response, HTTP error) returns an error
``Stimulus`` rather than raising — additive-plugin invariant per
CLAUDE.md.

Self-controlled args (per call): query, max_results, categories,
language, time_range, safesearch, engines. Operator-controlled args
(per-plugin config): instance_url, request_timeout_s, defaults for
all the per-call args.
"""
from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol

from krakey.interfaces.tool import Tool
from krakey.models.stimulus import Stimulus

if TYPE_CHECKING:
    from krakey.interfaces.plugin_context import PluginContext


MAX_RESULTS_CAP = 50
"""Hard ceiling on results-per-call. SearXNG aggregates multiple
engines so 20-50 fits one heartbeat without bloating the prompt; for
broader coverage Self should issue a refined query, not a wider one."""

DEFAULT_MAX_RESULTS = 10
"""Default count when Self omits ``max_results`` AND the per-plugin
config has no override."""

OUTPUT_BODY_CHARS = 240
"""Per-result snippet truncation cap. Same shape as
``duckduckgo_search`` so Self learns one number across providers."""

_VALID_TIME_RANGES: tuple[str, ...] = (
    "", "day", "week", "month", "year",
)
_VALID_SAFESEARCH: tuple[int, ...] = (0, 1, 2)


class SearxngBackend(Protocol):
    """Narrow Protocol the tool expects from any SearXNG client.

    Decouples the tool from ``aiohttp``: tests pass a dict-returning
    fake; production uses ``HttpSearxngClient`` from ``client.py``.

    ``max_results`` is keyword-only — the backend uses it to paginate
    SearXNG's ``/search?pageno=N`` until it has gathered enough
    results (SearXNG returns ~10 per page by default; without
    pagination Self could never reach the per-call cap). Filter args
    (``categories`` / ``language`` / ``time_range`` / ``safesearch``
    / ``engines``) mirror SearXNG's HTTP params; an empty / missing
    value means "skip this filter, let SearXNG use its default".
    """

    async def search(
        self, query: str, *, max_results: int, **filters: Any,
    ) -> dict[str, Any]: ...


class SearxngSearchTool(Tool):
    """Self-facing tool that runs one SearXNG query per call."""

    def __init__(
        self,
        backend: SearxngBackend,
        *,
        default_max_results: int = DEFAULT_MAX_RESULTS,
        default_categories: str = "",
        default_language: str = "",
        default_safesearch: int = 0,
    ):
        self._backend = backend
        self._default_max_results = default_max_results
        self._default_categories = default_categories
        self._default_language = default_language
        self._default_safesearch = default_safesearch

    @property
    def name(self) -> str:
        return "searxng_search"

    @property
    def description(self) -> str:
        return (
            "Web search via a local SearXNG aggregator (combines "
            "results across Google / Bing / DuckDuckGo / Wikipedia / "
            "etc). Only `query` is required — every other arg has a "
            "sensible default, so a minimal call like "
            "`{\"query\": \"python asyncio\"}` works out of the box. "
            "Override only what you need to. Use this tool when you "
            "want broader coverage than a single backend gives, or "
            "when you want to constrain by engine / category. "
            f"`max_results` (default {DEFAULT_MAX_RESULTS}, cap "
            f"{MAX_RESULTS_CAP}) controls breadth — the backend "
            "transparently paginates SearXNG's ~10-per-page response "
            "until your target count is reached (or SearXNG runs "
            "out). Optional filters: `categories` (csv: general,news,"
            "images,videos,music,files,it,science,...), `language` "
            "(e.g. 'en' or 'auto'), `time_range` ('day' | 'week' | "
            "'month' | 'year'), `safesearch` (0=off, 1=moderate, "
            "2=strict), `engines` (array of names like "
            "['google','bing']). Returns ranked results (title + "
            "url + snippet + originating engine) as a tool_feedback "
            "stimulus. Backend failures (instance unreachable, JSON "
            "disabled in SearXNG settings, etc.) surface as a clear "
            "error stim — issue a `cli_exec` to check the SearXNG "
            "container state if needed."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "required": ["query"],
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Search keywords. Falls back to the "
                        "natural-language intent if omitted."
                    ),
                },
                "max_results": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_RESULTS_CAP,
                    "description": (
                        "How many results to return. Scale up for "
                        "thoroughness, down for a quick check. Hard "
                        f"cap {MAX_RESULTS_CAP}; values above are "
                        "silently clamped."
                    ),
                },
                "categories": {
                    "type": "string",
                    "description": (
                        "Comma-separated SearXNG categories "
                        "(general, news, images, videos, music, "
                        "files, it, science, social_media, map, "
                        "...). Empty = SearXNG default."
                    ),
                },
                "language": {
                    "type": "string",
                    "description": (
                        "Language hint, e.g. 'en' or 'auto'. Empty "
                        "= no preference."
                    ),
                },
                "time_range": {
                    "type": "string",
                    "enum": list(_VALID_TIME_RANGES),
                    "description": (
                        "Recency filter: '' (any), 'day', 'week', "
                        "'month', 'year'."
                    ),
                },
                "safesearch": {
                    "type": "integer",
                    "enum": list(_VALID_SAFESEARCH),
                    "description": (
                        "0 = off, 1 = moderate, 2 = strict."
                    ),
                },
                "engines": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Specific engines to query "
                        "(e.g. ['google', 'duckduckgo', 'bing']). "
                        "Empty / omitted = SearXNG decides from "
                        "categories."
                    ),
                },
            },
        }

    async def execute(
        self, intent: str, params: dict[str, Any],
    ) -> Stimulus:
        # ---- query ----
        query = (params.get("query") or intent or "").strip()
        if not query:
            return self._err(
                "missing `query` (and no fallback intent provided)",
            )

        # ---- max_results ----
        raw = params.get("max_results")
        if raw is None:
            max_results = self._default_max_results
        else:
            # bool is a subclass of int — exclude it explicitly so
            # ``max_results: True`` doesn't silently coerce to 1.
            if isinstance(raw, bool) or not isinstance(raw, int):
                return self._err(
                    "`max_results` must be a positive integer",
                )
            if raw < 1:
                return self._err(
                    "`max_results` must be >= 1",
                )
            # Soft-cap: clamp instead of error so Self learns the
            # ceiling without us aborting her call.
            max_results = min(raw, MAX_RESULTS_CAP)

        # ---- categories ----
        categories = params.get("categories", self._default_categories)
        if not isinstance(categories, str):
            return self._err("`categories` must be a string (csv)")

        # ---- language ----
        language = params.get("language", self._default_language)
        if not isinstance(language, str):
            return self._err("`language` must be a string")

        # ---- time_range ----
        time_range = params.get("time_range", "")
        if time_range not in _VALID_TIME_RANGES:
            return self._err(
                "`time_range` must be one of "
                f"{list(_VALID_TIME_RANGES)}",
            )

        # ---- safesearch ----
        safesearch = params.get(
            "safesearch", self._default_safesearch,
        )
        if isinstance(safesearch, bool) or not isinstance(
            safesearch, int,
        ):
            return self._err(
                "`safesearch` must be an integer in "
                f"{list(_VALID_SAFESEARCH)}",
            )
        if safesearch not in _VALID_SAFESEARCH:
            return self._err(
                "`safesearch` must be one of "
                f"{list(_VALID_SAFESEARCH)}",
            )

        # ---- engines ----
        engines_raw = params.get("engines")
        if engines_raw is None:
            engines_csv = ""
        else:
            if (
                not isinstance(engines_raw, list)
                or not all(isinstance(e, str) for e in engines_raw)
            ):
                return self._err(
                    "`engines` must be an array of strings",
                )
            engines_csv = ",".join(engines_raw)

        # ---- dispatch ----
        # ``max_results`` is threaded through so the backend can
        # paginate (SearXNG returns ~10 results per page; without
        # pagination Self could never reach the per-call cap).
        try:
            payload = await self._backend.search(
                query,
                max_results=max_results,
                categories=categories,
                language=language,
                time_range=time_range,
                safesearch=safesearch,
                engines=engines_csv,
            )
        except Exception as e:  # noqa: BLE001
            return self._err(f"backend error: {e}")

        results = (payload or {}).get("results") or []
        if not results:
            return self._stim(
                f"No results for query: {query!r}.",
            )
        # Defensive slice — backend should already cap, but a
        # buggy / future backend returning more shouldn't pollute
        # Self's prompt.
        results = results[:max_results]
        return self._stim(_format(results, query))

    def _stim(self, content: str) -> Stimulus:
        return Stimulus(
            type="tool_feedback",
            source=f"tool:{self.name}",
            content=content,
            timestamp=datetime.now(),
            adrenalin=False,
        )

    def _err(self, msg: str) -> Stimulus:
        return Stimulus(
            type="tool_feedback",
            source=f"tool:{self.name}",
            content=f"searxng_search error: {msg}",
            timestamp=datetime.now(),
            adrenalin=False,
        )


def _format(results: list[dict[str, Any]], query: str) -> str:
    """Render the SearXNG results list to Self's prompt-friendly form.

    Format mirrors ``duckduckgo_search``'s output (title + url +
    optional snippet) but appends the originating engine in ``[brackets]``
    so Self can compare which engines surfaced which result — useful
    for cross-source verification.
    """
    lines = [f"SearXNG results for {query!r} — {len(results)}:"]
    for i, r in enumerate(results, 1):
        title = (r.get("title") or "(untitled)").strip()
        url = (r.get("url") or "").strip()
        body = (r.get("content") or "").strip().replace("\n", " ")
        engine = (r.get("engine") or "").strip()
        if len(body) > OUTPUT_BODY_CHARS:
            body = body[: OUTPUT_BODY_CHARS - 1] + "…"
        suffix = f" [{engine}]" if engine else ""
        lines.append(f"{i}. [{title}] {url}{suffix}")
        if body:
            lines.append(f"    {body}")
    return "\n".join(lines)


def build_tool(ctx: "PluginContext") -> "SearxngSearchTool":
    """Factory invoked by ``load_component``. Resolves config,
    optionally auto-starts a Docker container, and constructs the
    HTTP client + tool.

    Imports of ``client``/``lifecycle`` are local so the tool module
    stays importable for unit tests even when ``aiohttp`` / ``docker``
    are absent — tests construct ``SearxngSearchTool`` directly with
    a fake backend.
    """
    from krakey.plugins.searxng_search.client import HttpSearxngClient
    from krakey.plugins.searxng_search.lifecycle import (
        ensure_instance_running,
    )

    cfg = ctx.config or {}
    instance_url = str(
        cfg.get("instance_url") or "http://127.0.0.1:8888",
    ).rstrip("/")
    timeout_s = float(cfg.get("request_timeout_s") or 15)

    if bool(cfg.get("auto_start", False)):
        # Best-effort: failure logs but does not raise — the tool's
        # per-call HTTP error handling surfaces "connection refused"
        # to Self with a clear message.
        ensure_instance_running(
            instance_url=instance_url,
            docker_image=str(
                cfg.get("docker_image") or "searxng/searxng:latest",
            ),
            container_name=str(
                cfg.get("container_name") or "krakey-searxng",
            ),
            host_port=int(cfg.get("host_port") or 8888),
        )

    backend = HttpSearxngClient(
        instance_url=instance_url, timeout_s=timeout_s,
    )
    return SearxngSearchTool(
        backend=backend,
        default_max_results=int(
            cfg.get("default_max_results") or DEFAULT_MAX_RESULTS,
        ),
        default_categories=str(cfg.get("default_categories") or ""),
        default_language=str(cfg.get("default_language") or ""),
        default_safesearch=int(cfg.get("default_safesearch") or 0),
    )
