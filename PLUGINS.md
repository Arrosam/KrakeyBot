# KrakeyBot Plugin Development

Plugins extend Krakey by contributing one or more **components**:

| Component | Purpose | Examples |
|---|---|---|
| `reflect` | Heartbeat hook — claims a role string the runtime queries by | hypothalamus, recall_anchor, in_mind |
| `tentacle` | Outbound action Self can dispatch | search, telegram_reply, update_in_mind |
| `sensory`  | Inbound stimulus producer | telegram, dashboard web chat |

Plugins are **strictly additive**: disabling or removing any plugin must
not break the runtime's core loop. Every plugin call site has a
graceful fallback (null object, soft fail, phase skip). Don't introduce
code that requires a plugin to be present.

## Layout

One folder per plugin under either root:

| Root | Owner |
|---|---|
| `src/plugins/<name>/` | ships with Krakey (built-in) |
| `workspace/plugins/<name>/` | user-installed |

Each folder contains:

- `meta.yaml` (required) — manifest read by both the runtime loader and
  the dashboard catalogue scanner without importing plugin code.
- Component code (`reflect.py`, `tentacle.py`, `sensory.py`, or a flat
  `__init__.py`) exposing one factory per component.

User-editable per-plugin config goes in `workspace/plugins/<name>/config.yaml`
regardless of whether the plugin code is built-in or workspace.

## meta.yaml

The full schema lives in [`src/plugin_system/loader.py`](src/plugin_system/loader.py).
Minimal example:

```yaml
name: my_plugin
description: "..."
config_schema: []          # plugin-level config fields (UI hints only)
requires_sandbox: false    # set true if any component touches the sandbox VM

components:
  - kind: reflect
    role: my_role          # required for kind=reflect; must be unique
    factory_module: src.plugins.my_plugin.reflect
    factory_attr: build_reflect
    llm_purposes:          # optional
      - name: translator
        description: "..."
        suggested_tag: fast_generation

  - kind: tentacle
    factory_module: src.plugins.my_plugin.tentacle
    factory_attr: build_tentacle

  - kind: sensory
    factory_module: src.plugins.my_plugin.sensory
    factory_attr: build_sensory
```

A plugin can ship any combination of components. Enabling a plugin in
`config.yaml`'s `plugins:` list loads **all** its components; there is
no per-component toggle.

## Factory contract

Every component factory has the same signature:

```python
def build_<kind>(ctx: PluginContext) -> ComponentInstance: ...
```

Returning `None` opts out (e.g. when a required LLM purpose isn't bound)
without crashing the runtime. The full `PluginContext` surface (config
view, services whitelist, plugin_cache for multi-component plugins,
LLM resolution) is documented in
[`src/interfaces/plugin_context.py`](src/interfaces/plugin_context.py).

## Enabling a plugin

In the central `config.yaml`:

```yaml
plugins:
  - my_plugin
  - duckduckgo_search
  - hypothalamus
```

All plugins default OFF. Empty list = zero components. Order in the
list determines registration order for plugins that contribute to the
same registry.

## Examples in-tree

- **Single-component tentacle**: [`src/plugins/duckduckgo_search/`](src/plugins/duckduckgo_search/)
  — flat `__init__.py` with backend Protocol, tentacle, factory.
- **Multi-component sharing a client**: [`src/plugins/telegram/`](src/plugins/telegram/)
  — sensory + tentacle share an `HttpTelegramClient` via `ctx.plugin_cache`.
- **Multi-component sharing a reflect**: [`src/plugins/in_mind_note/`](src/plugins/in_mind_note/)
  — reflect owns state, tentacle mutates it; both wired through
  `ctx.plugin_cache`.
- **Multi-component as a pipeline**: [`src/plugins/recall/`](src/plugins/recall/)
  — passive reflect (auto-recall, claims `recall_anchor` role) +
  active tentacle (`memory_recall`, Self drills into noticed nodes).
  Two halves of one discovery flow; ship together so the pipeline
  enables/disables atomically. Shared GM-query primitive in
  `gm_query.py`.

## Architectural rules

1. **No code load before user enable.** The loader walks meta.yaml files
   only; plugin Python modules are imported lazily on enable via
   `load_component(component, ctx)`.
2. **Plugin granularity for enable.** Listing a plugin loads all its
   components together.
3. **All plugins default off.** The user must explicitly opt in.
4. **Plugins never crash startup.** Bad config → log warning + skip.
   Missing LLM purpose → factory returns None → component absent.
   Architectural invariant per `CLAUDE.md`.
