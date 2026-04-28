# Project conventions for Claude Code

## 🔒 Core architectural invariant — plugins are strictly additive

**Disabling or removing ANY plugin (Reflects, tools, sensories)
must NOT break the runtime's core loop.** Set 2026-04-25 by Samuel.

This is load-bearing. Every plugin call site in `src/` must have a
graceful fallback — null-object (e.g. `NoopRecall`), soft-fail (e.g.
tool dispatch produces a `Unknown tool: X` system event for
Self instead of raising), or a phase skip. **No `raise RuntimeError`
when a plugin is missing.**

Test the invariant: `tests/test_zero_plugin_runtime.py` runs Runtime
with all Reflects unregistered + zero tools + zero sensories and
asserts it completes a heartbeat. New code that requires a plugin
to function will break that test — fix the missing-plugin path
before merging.

UX defaults (which Reflects auto-register, what tools ship
in-tree) are a separate concern from this invariant. Default-on
plugins exist for convenience; the architecture must work with
ALL of them disabled.

## Architecture documentation

The codebase has one always-fresh, interactive view, served live:

```
python docs/scripts/serve_arch_graph.py
# → http://127.0.0.1:8979/
```

The server walks `src/`, builds a Cytoscape.js dependency graph
(folders → files → classes → methods, with import + reference edges)
in-memory, and pushes updates over Server-Sent Events whenever any
file under `src/` changes. The page hot-swaps elements in place, so
expand/collapse, right-click hide, pan, and zoom state survive
across rebuilds.

Interactions: double-click a compound to expand/collapse, single-click
to inspect (panel + neighborhood highlight), right-click to hide a
subtree at 20% opacity, hover for the docstring.

The implementation lives in `docs/scripts/`
(`serve_arch_graph.py` + the `build_arch_graph.py` library it imports).

## Commit messages

- No `Co-Authored-By: Claude` trailer on commits.
- Follow the existing commit style in this repo: short imperative
  subject, explanatory body describing the "why".
