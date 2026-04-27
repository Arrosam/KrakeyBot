# KrakeyBot Plugin Development

Krakey discovers plugins at boot by scanning two roots:

| Root | Source tag | Who owns it |
|---|---|---|
| `src/plugins/builtin/` | `builtin` | ships with Krakey |
| `workspace/plugins/`   | `plugin`  | user-dropped |

Everything under those roots is organised as **projects**. One project =
one folder. A project may contain one tentacle, one sensory, or a
bundle of both that share state (e.g. a Telegram project that ships a
sensory + a reply tentacle sharing one HTTP client).

## Quick rules

- **Plugins run in-process with Krakey's privileges.** Use the sandbox
  VM (`SANDBOX.md`) for anything untrusted.
- **A bad plugin must never block boot.** The loader catches every
  exception, reports it on the dashboard (red card), and moves on.
- **Per-project config lives under `plugins.<project_name>`** in
  `config.yaml`. There is no separate `tentacle:` / `sensory:`
  section — one project gets one dict.

## Layout

Two equally valid shapes:

### Single file (simplest)

```
workspace/plugins/my_weather.py
```

### Package (helper modules, data files, manifest.yaml)

```
workspace/plugins/my_weather/
  __init__.py
  helpers.py
  manifest.yaml        # optional — wins over inline MANIFEST
```

Rules:

- Subdirs / files starting with `.` or `_`, and `__pycache__`, are skipped.
- Non-`.py` files at the root (like a stray README) are ignored.

## Contract

Your project module must expose **one** of three factories. Pick by
what the project produces:

### A. Multi-component project — `create_plugins`

Recommended when the project produces more than one thing, especially
if they share state (one client, one cache, one connection pool, …).

```python
def create_plugins(config: dict, deps: dict) -> dict:
    client = HttpClient(token=config["bot_token"])
    return {
        "tentacles": [MyReplyTentacle(client)],
        "sensories": [MyPollSensory(client)],
    }
```

Return shape: `{"tentacles": [...], "sensories": [...]}`. Either key
may be empty or omitted.

### B. Single-tentacle shortcut — `create_tentacle`

```python
def create_tentacle(config: dict, deps: dict) -> Tentacle:
    return WeatherTentacle(api_key=config["api_key"])
```

### C. Single-sensory shortcut — `create_sensory`

```python
def create_sensory(config: dict, deps: dict) -> Sensory:
    return FileSystemWatchSensory(watch_dir=config["dir"])
```

### D. Bare-class fallback

If the module exposes no factory, the loader looks for any subclass
of `Tentacle` or `Sensory` and instantiates it with `cls(**config)` or
`cls()`. Useful for trivial zero-config plugins.

## `deps` — runtime-injected dependencies

All factories receive the same `deps` dict:

| Key | Type | Notes |
|---|---|---|
| `gm` | `GraphMemory` | the live graph |
| `kb_registry` | `KBRegistry` | long-term memories |
| `embedder` | `AsyncEmbedder` | `await embedder(text) → vec` |
| `buffer` | `StimulusQueue` | push stimuli back at Krakey |
| `events` | `EventBus` | publish typed events (dashboard subscribers consume) |
| `runtime` | `Runtime` \| None | full runtime ref — used by the dashboard plugin to wire its server adapters; ordinary plugins should not need this |
| `config` | full `Config` object | read-only view of the rest of the config |
| `build_code_runner` | callable | `(coding_cfg) → CodeRunner` — honours sandbox policy; used by the `coding` plugin |

## Manifest

Inline in the module:

```python
MANIFEST = {
    "name": "my_weather",               # defaults to folder name
    "description": "Current weather for a city",
    "is_internal": False,                # single-tentacle shortcut: default is_internal
    "components": [                      # REQUIRED for create_plugins
        {"kind": "tentacle", "name": "weather",
         "is_internal": True,
         "description": "inward: results go to Self"},
        {"kind": "sensory", "name": "weather_alerts",
         "description": "optional paired sensory"},
    ],
    "config_schema": [
        # DO NOT declare `enabled` — it is a reserved key owned by the
        # loader (default False). The dashboard renders a dedicated
        # toggle for it above your rows.
        {"field": "api_key",     "type": "password", "default": "",
         "help": "openweathermap.org key"},
        {"field": "default_city","type": "text",     "default": "Auckland"},
    ],
}
```

Or as `manifest.yaml` next to the module (YAML overrides inline
`MANIFEST` field-for-field):

```yaml
name: my_weather
description: Current weather for a city
components:
  - kind: tentacle
    name: weather
    is_internal: true
config_schema:
  # `enabled` is reserved — do NOT add it here.
  - field: api_key
    type: password
    default: ""
    help: openweathermap.org key
```

### Reserved keys

- `enabled` — loader-owned. Default **`False`**. The factory never runs
  until the user sets `plugins.<project>.enabled: true` in
  `config.yaml` (or toggles it on in the dashboard). Any `enabled`
  entry a plugin author writes into `config_schema` is stripped
  silently on load.

### Component metadata

For multi-component projects the `components` list declares each
produced tentacle / sensory:

- `kind` — `"tentacle"` or `"sensory"` (required)
- `name` — component name as it will appear in `[STATUS]` and in
  the dashboard (defaults to the instance's `.name` property)
- `description` — one-liner shown in the dashboard card
- `is_internal` — tentacles only; overrides the instance's own
  `is_internal` property

### Field types the dashboard understands

| `type` | Widget |
|---|---|
| `bool` | toggle |
| `number` | integer input (no spinner) |
| `number_float` | decimal input |
| `text` | single-line text input |
| `password` | masked text input |

Unrecognised types fall back to `text`.

## Config lookup

The loader reads `config.yaml`'s `plugins.<project_name>` entry and
passes it to the factory. For a `my_weather` project:

```yaml
plugins:
  my_weather:
    enabled: true
    api_key: "${OPENWEATHER_API_KEY}"
    default_city: Wellington
```

Defaults from `config_schema` fill missing keys before the factory
sees the dict. `enabled` defaults to `false`; until the user explicitly
sets it to `true`, the project is reported in the dashboard but its
module is imported only for manifest reading — the factory never runs.

## Examples

### Minimal single-tentacle plugin

```python
# workspace/plugins/echo.py
from datetime import datetime

from src.interfaces.tentacle import Tentacle
from src.models.stimulus import Stimulus


MANIFEST = {
    "description": "Echoes the intent back to Self as tentacle_feedback.",
    "is_internal": True,
    "config_schema": [
        {"field": "prefix",  "type": "text", "default": "echo: "},
    ],
}


class EchoTentacle(Tentacle):
    def __init__(self, prefix):
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
            type="tentacle_feedback", source="tentacle:echo",
            content=self._prefix + (intent or ""),
            timestamp=datetime.now(), adrenalin=False,
        )


def create_tentacle(config, deps):
    return EchoTentacle(prefix=config.get("prefix", "echo: "))
```

Enable in `config.yaml`:

```yaml
plugins:
  echo:
    enabled: true
    prefix: "→ "
```

### Multi-component project sharing a client

See `src/plugins/builtin/telegram/__init__.py` — the canonical example:
sensory + reply tentacle sharing one `HttpTelegramClient` via
`create_plugins`.

## Troubleshooting

- **Plugin does not show up** — check that `workspace/plugins/<name>/`
  (or `workspace/plugins/<name>.py`) exists and is not hidden
  (no leading `.` / `_`). Check the startup log for
  `plugin ... failed to load`.
- **Plugin shows `plugin ✗` in dashboard** — expand the card; the
  full traceback is shown. Common causes: missing host dependency,
  typo in factory name, factory raises during build.
- **Plugin loads but Krakey never calls it** — the Hypothalamus picks
  tentacles from `[STATUS]`. Check that `name` + `description` are
  descriptive enough for Self to map her intent onto.
- **Config changes ignored** — plugin instances are built at Runtime
  construction. Restart Krakey after editing `config.yaml` (or save
  via dashboard Settings + click Restart).
