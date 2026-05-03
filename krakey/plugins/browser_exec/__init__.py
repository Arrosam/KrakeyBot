"""``browser_exec`` plugin — single tool that runs a real browser
instance inside a target Environment via Playwright.

Companion to ``cli_exec`` and ``gui_exec``: same architecture
(Self picks env per call, plugin must be allow-listed in the
Environment Router). Each tool call is one *session script*: Self
sends ``start_url`` + an ordered list of typed actions
(navigate / click / type / press / scroll / wait_for / screenshot),
all executed inside one Playwright session, browser opens at the
start of the call and closes at the end. No state survives across
calls — to resume from a post-click page on the next call, Self
reads ``final_url`` from the previous response and passes it as
``start_url``.

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
    ``python3`` on minimal Linux installs.

A missing dep surfaces as a non-zero rc with ``ModuleNotFoundError``
in stderr; Self gets an error stimulus and reports it instead of
the runtime crashing — the additive-plugin invariant holds.

Manual smoke (operator-only, not automated): with the plugin
allow-listed for ``local`` and Playwright + Chromium installed,
have Self emit::

    <tool_call>
    {"name": "browser_exec",
     "arguments": {"env": "local",
                   "start_url": "https://example.com",
                   "actions": []}}
    </tool_call>

and confirm the next-beat tool_feedback Stimulus carries an a11y
tree rooted at ``Example Domain``.
"""
