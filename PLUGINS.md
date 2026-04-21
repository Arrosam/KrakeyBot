# KrakeyBot Plugin Development

Krakey discovers extra tentacles and sensories at boot by scanning
`workspace/tentacles/` and `workspace/sensories/`. Anything you drop
there (with a valid module layout) is registered alongside the
built-ins and appears in the dashboard's `Plugins (auto-discovered)`
section.

## Quick rules

- **Plugins run in-process with Krakey's privileges.** A plugin that
  imports `os` can do everything your shell user can. Use the sandbox
  VM (`SANDBOX.md`) for anything untrusted.
- **A bad plugin must not block boot.** The loader catches every
  exception, logs it, reports on the dashboard, and moves on.
- **Per-plugin config lives under `config.yaml`** at
  `tentacle.<name>` / `sensory.<name>` — same shape as the built-ins.
  Plugins declare their schema in a manifest so the dashboard can
  render typed inputs.

## Layout

Two equally valid shapes:

**Single file** (simplest):

```
workspace/tentacles/my_weather.py
```

**Package** (if you need helper modules / data files):

```
workspace/tentacles/my_weather/
  __init__.py
  helpers.py
  manifest.yaml        # optional — overrides / supplements inline MANIFEST
```

Rules:

- Subdirs starting with `.` or `_` and `__pycache__` are skipped.
- The same rule applies to `workspace/sensories/`.

## Contract

A plugin module must expose **one** of:

### A. Factory function (recommended)

```python
# workspace/tentacles/my_weather.py

def create_tentacle(config: dict, deps: dict) -> "Tentacle":
    return MyWeatherTentacle(
        api_key=config["api_key"],
        city=config.get("default_city", "Auckland"),
    )
```

`config` is the per-plugin slice of `config.yaml`
(`tentacle.my_weather` merged with defaults from your manifest).

`deps` gives you access to runtime primitives you may need:

| Key | Type | Notes |
|---|---|---|
| `gm` | `GraphMemory` | the live graph |
| `kb_registry` | `KBRegistry` | long-term memories |
| `embedder` | `AsyncEmbedder` | `await embedder(text) → vec` |
| `buffer` | `StimulusBuffer` | push stimuli back at Krakey |
| `web_chat_history` | `WebChatHistory` or `None` | dashboard chat |
| `config` | full `Config` object | read-only views of other sections |

Return a `Tentacle` (or `Sensory`) instance — see `src/interfaces/`.

Sensory variant: factory is `create_sensory(config, deps) -> Sensory`.

### B. Exported class

If your class needs no dependencies and its `__init__` takes only
kwargs:

```python
class WeatherTentacle(Tentacle):
    def __init__(self, api_key: str = "", **_):
        ...

TENTACLE_CLASS = WeatherTentacle
```

The loader will call `TENTACLE_CLASS(**config)`. Use `SENSORY_CLASS`
for sensories.

### C. Bare subclass

If neither factory nor class constant is exported, the loader looks
for any subclass of `Tentacle` / `Sensory` in the module and uses it.
Suitable for one-file plugins with zero config.

## Manifest

Either inline in the module:

```python
MANIFEST = {
    "name": "my_weather",                  # defaults to filename / dir name
    "description": "Current weather for a city",
    "is_internal": False,                   # tentacles only — see below
    "config_schema": [
        {"field": "enabled",       "type": "bool",    "default": True,
         "help": "Master switch"},
        {"field": "api_key",       "type": "password","default": "",
         "help": "openweathermap.org API key"},
        {"field": "default_city",  "type": "text",    "default": "Auckland"},
        {"field": "max_age_min",   "type": "number",  "default": 30,
         "help": "Cache TTL in minutes"},
    ],
}
```

or as `manifest.yaml` next to the module:

```yaml
name: my_weather
description: Current weather for a city
is_internal: false
config_schema:
  - field: enabled
    type: bool
    default: true
  - field: api_key
    type: password
    default: ""
    help: openweathermap.org API key
```

YAML wins over inline when both are present.

### Field types the dashboard understands

| `type` | Widget |
|---|---|
| `bool` | toggle |
| `number` | integer input (no spinner) |
| `number_float` | decimal input |
| `text` | single-line text input |
| `password` | masked text input |

Anything else falls back to `text`.

### `is_internal` — what it means

`is_internal=true` means the tentacle's output is Krakey's own private
information (e.g. `memory_recall`, `search`). The runtime logs it in
magenta and Self decides whether to relay the content to the human.

`is_internal=false` means the tentacle's output goes directly to a
real human (e.g. `web_chat_reply`). Runtime logs it in green as
Krakey's outward voice.

Sensories are always "inward" — they feed stimuli, they never emit
user-visible text.

## Enabling / disabling

The loader checks `config[tentacle_or_sensory][name].enabled`. If
false, the plugin is discovered and listed in the dashboard but NOT
registered on the runtime registry. This lets the dashboard report
"present but disabled" without edit friction.

Default when the field is absent: `true`.

## Examples

A minimal "echo" tentacle that just returns its intent as an inward
stimulus — good template to copy:

```python
# workspace/tentacles/echo.py
from datetime import datetime

from src.interfaces.tentacle import Tentacle
from src.models.stimulus import Stimulus


MANIFEST = {
    "description": "Echoes the intent back to Self as a tentacle_feedback.",
    "is_internal": True,
    "config_schema": [
        {"field": "enabled", "type": "bool", "default": True},
        {"field": "prefix",  "type": "text", "default": "echo: "},
    ],
}


class EchoTentacle(Tentacle):
    def __init__(self, prefix: str = "echo: ", **_):
        self._prefix = prefix

    @property
    def name(self): return "echo"
    @property
    def description(self): return MANIFEST["description"]
    @property
    def parameters_schema(self):
        return {"intent": "free text to echo back"}
    @property
    def is_internal(self): return True

    async def execute(self, intent, params):
        return Stimulus(
            type="tentacle_feedback",
            source=f"tentacle:{self.name}",
            content=self._prefix + (intent or ""),
            timestamp=datetime.now(),
            adrenalin=False,
        )


def create_tentacle(config, deps):
    return EchoTentacle(prefix=config.get("prefix", "echo: "))
```

Drop that file in `workspace/tentacles/echo.py`, add
```yaml
tentacle:
  echo: { enabled: true, prefix: "→ " }
```
to `config.yaml`, and restart. The dashboard's Plugins section will
show it with a `plugin ✓` badge.

## Troubleshooting

- **Plugin does not show up** — check `workspace/tentacles/<name>/`
  exists and is not hidden (starts with `.` / `_`). Check the runtime
  log at startup for `plugin ... failed to load`.
- **Plugin shows as `plugin ✗`** — hover / expand the card in the
  dashboard Plugins section; the full traceback is shown. Common
  causes: missing dependency on the host, typo in `create_tentacle`
  name, factory raising in its own init.
- **Plugin loads but Krakey never calls it** — Hypothalamus picks
  tentacles from `[STATUS]`. Check that `description` + name are
  something Self can map to her intent. Rename to something less
  generic if needed.
- **Config changes ignored** — plugin instances are built at Runtime
  construction. Restart Krakey after editing `config.yaml` (or save
  via dashboard Settings + click Restart).
