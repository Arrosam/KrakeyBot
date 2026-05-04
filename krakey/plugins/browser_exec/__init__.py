"""``browser_exec`` plugin — single tool that drives a *persistent*
Playwright browser instance inside a target Environment.

Companion to ``cli_exec`` and ``gui_exec``: same architecture
(Self picks env per call, plugin must be allow-listed in the
Environment Router). Unlike its siblings, ``browser_exec`` runs a
**long-running browser RPC server inside the env** that survives
across heartbeats. The first tool call spawns the server detached;
subsequent calls reach it over a localhost TCP loopback (inside
the env). The browser instance, all tabs, their DOM state and
in-memory JS state, and any cookies persist for the lifetime of
the env (host process for ``local``, guest VM for ``sandbox``).

Tool surface — one tab per call, picked by top-level ``action``:

  * ``action: "list_tabs"`` — read the live tab map.
  * ``action: "new_tab"`` — open a new tab, navigate to
    ``start_url``, return its tab_id.
  * ``action: "close_tab"`` — close a specific tab by ``tab_id``.
  * ``action: "operate"`` — run an in-tab action chain (navigate /
    click / type / press / scroll / wait_for / screenshot) on a
    specific tab. The browser instance and the tab's DOM/JS
    state survive across calls — use ``operate`` repeatedly on
    the same ``tab_id`` to do multi-step flows.

Every successful response includes the current ``tabs`` list so
Self always knows what's open (mirrors ``in_mind_note``'s pattern
of injecting state into the prompt).

Default extraction is the page's accessibility tree
(``page.accessibility.snapshot()``) — semantic, structurally
faithful, far token-cheaper than raw HTML. ``output: "text"`` and
``"html"`` modes are also exposed.

Operator setup (the plugin does NOT provision either):

  * The env's Python interpreter must have ``playwright`` installed
    (``pip install playwright``) and at least one bundled browser
    downloaded (``playwright install chromium``, ~200MB per
    browser).
  * The plugin must be allow-listed in the Environment Router,
    e.g.::

        environments:
          local:
            allowed_plugins: [..., browser_exec]

  * The interpreter name is configurable via the plugin's
    ``python_cmd`` config field (default ``"python"``); set to
    ``python3`` on minimal Linux installs that ship only
    ``python3``.

A missing dep surfaces in the dispatched server's logs and the
tool's response carries a tail of the log so the operator can
fix it. The runtime keeps heartbeating — additive-plugin
invariant holds.

Crash semantics: if the in-env server crashes, all tabs are lost
(state is in-memory only). Self sees an empty tab list on the
next call and re-opens what she needs. Persisting tab state to
disk for crash recovery is deferred until needed.

Manual smoke (operator-only, not automated): with the plugin
allow-listed for ``local`` and Playwright + Chromium installed,
have Self emit::

    <tool_call>
    {"name": "browser_exec",
     "arguments": {"env": "local",
                   "action": "new_tab",
                   "start_url": "https://example.com",
                   "label": "smoke"}}
    </tool_call>

and confirm the next-beat tool_feedback Stimulus carries an
``opened tab_id='tab_...'`` line plus the tab list. Then::

    <tool_call>
    {"name": "browser_exec",
     "arguments": {"env": "local",
                   "action": "operate",
                   "tab_id": "<from above>",
                   "actions": [],
                   "output": "a11y"}}
    </tool_call>

and confirm the a11y tree comes back rooted at ``Example Domain``,
proving the persistent browser is reachable across calls.
"""
