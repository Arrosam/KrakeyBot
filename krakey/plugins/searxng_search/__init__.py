"""``searxng_search`` plugin — web search via a local SearXNG instance.

SearXNG is a metasearch aggregator: one query → results merged across
many backends (Google / Bing / DuckDuckGo / Wikipedia / etc.). This
plugin queries a SearXNG instance over HTTP and returns the merged
results to Self as a ``tool_feedback`` Stimulus.

Operator setup (the plugin does NOT pre-provision SearXNG):

  1. Run a SearXNG instance on the configured ``instance_url``. The
     reference path is Docker:

         docker run -d --rm \\
             --name krakey-searxng -p 8888:8080 \\
             searxng/searxng:latest

     Set ``auto_start: true`` (per-plugin config) and the plugin
     will run that command itself when it sees no instance on
     ``instance_url`` at first tool call. Idempotent — a second
     start with a healthy instance is a no-op.

  2. SearXNG's ``settings.yml`` must enable JSON output
     (``search.formats: [json, html]``). The official Docker image
     ships JSON disabled by default; mount a custom settings file
     or set ``SEARXNG_SETTINGS_PATH`` accordingly. Without JSON
     enabled the plugin's per-call HTTP request will return 403
     and surface as ``backend error: 403`` to Self.

Why this AND ``duckduckgo_search``: DDG is single-backend + zero-ops
(no instance to run), this is multi-backend + operator-controlled
(pick engines, categories, language, run private). Both can ship at
once — Self picks per call.

Tool name pin: ``searxng_search`` (NOT the abstract ``search``) so
the two plugins coexist in one runtime without a registry collision
— the registry rejects duplicate tool names. Self addresses this
plugin by its full name in ``<tool_call>``.
"""
