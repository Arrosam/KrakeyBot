# KrakeyBot

> An autonomous cognitive agent that maintains "presence" through a continuous heartbeat.
> See [`SANDBOX.md`](SANDBOX.md) for sandbox VM configuration (required reading before enabling any of the coding / GUI / file / browser tools).
> See [`PLUGINS.md`](PLUGINS.md) for custom tools / channels (drop a plugin project into `workspace/plugins/<project>/` and it auto-loads).

---

## Requirements

- **Python**: 3.12 (3.11 also supported)
- **LLM Server (chat)**: OpenAI-compatible `/v1/chat/completions`
  - Local: `llama-server` / `llama.cpp` / `vllm` / `lmstudio` / `ollama` (in OpenAI-compatible mode)
  - Cloud: DashScope / Anthropic Claude / One API aggregator proxies, etc.
- **Embedding Server**: OpenAI-compatible `/v1/embeddings` (required since Phase 1; `bge-m3` recommended)
- **Reranker Server (optional)**: OpenAI-compatible `/v1/rerank` (auto-falls-back to scripted ranking if absent; `bge-reranker-v2-m3` recommended)
- `sqlite-vec` is installed automatically by `pip install` (prebuilt wheels for Windows / Linux / macOS)

---

## Installation

### Users — install from PyPI

```bash
python -m venv .venv

# Linux / macOS
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

pip install krakey
```

After installation the `krakey` command is ready to use:

```bash
krakey --version          # krakey 0.1.0
krakey                    # show help
krakey onboard            # run the configuration wizard
krakey run                # start the heartbeat
```

Upgrade: `pip install -U krakey`.

### Developers — editable install from source

```bash
git clone https://github.com/Arrosam/KrakeyBot.git
cd KrakeyBot
python -m venv .venv && source .venv/bin/activate   # mac/linux
pip install -e ".[dev]"                              # includes pytest etc.
pytest -q                                            # 560+ tests, should all pass
```

In editable mode, `krakey update` / `krakey repair` operate on the
git tags of the local repository. PyPI users running `krakey update`
get a hint to use `pip install -U krakey` instead.

---

## Configuration

**First install: run the onboarding wizard** to generate `config.yaml`:

```bash
krakey onboard
```

The wizard walks you through three steps:
1. Pick a chat LLM provider (label / base URL / API key / model name); auto-bound to the `self_thinking` + `compact` + `classifier` core purposes.
2. Optional: configure an embedding provider/model (required by recall + KB indexing).
3. Tick the plugins to enable (**dashboard is ticked by default and strongly recommended** — without it there is no in-app way to see Krakey's state).

The wizard can be re-run at any time: an existing `config.yaml` is
backed up to `workspace/backups/` before being overwritten.

**Cloud APIs**: put the key in an environment variable, then reference
it from `config.yaml` with a `${ENV_VAR}` placeholder:
```bash
# Windows (PowerShell)
$env:DASHSCOPE_API_KEY = "sk-..."
# Linux/macOS
export DASHSCOPE_API_KEY=sk-...
```
In `config.yaml` write `api_key: ${DASHSCOPE_API_KEY}`; it is resolved
on load.

**Heartbeat / fatigue parameters** (optional fine-tuning; the
defaults are usually enough):
```yaml
hibernate:
  default_interval: 30   # default seconds to hibernate when idle
  min_interval: 2
  max_interval: 300
```

---

## Running

The `krakey` CLI offers two run modes:

```bash
krakey run        # foreground (terminal-attached, Ctrl+C to exit)
krakey start      # background daemon (detach, write pidfile + logs)
krakey stop       # stop the background process
krakey status     # query current state (running / stopped / version / log path)
```

In background mode:
- pidfile: `workspace/.krakey.pid`
- log: `workspace/logs/daemon.log`
- The daemon handles `SIGTERM` for graceful shutdown; force-killed after a 10s timeout.

Once started, the program enters its heartbeat loop:
- When idle, it sleeps for `default_interval` seconds (default 30s).
- Pressing Enter on terminal input (only in `krakey run` mode) → marked as an adrenalin stimulus → **interrupts hibernate immediately and wakes Self**.
- Self's decision is translated by the Hypothalamus → the `Action` tool is dispatched.
- Tool replies are printed to the terminal with the `[action] ...` prefix.
- `Ctrl+C` (or `krakey stop` in background mode) terminates the program.

### Example session

```
$ krakey run
[HB #1] stimuli=0 (thinking...)
[HB #1] decision: (none)
[HB #1] hibernate 10s
hello                                              # ← your input (typed at any time, wakes immediately)
[HB #2] stimuli=1 (thinking...)
[HB #2] decision: Use action tool to greet user.
[dispatch] action ← 'Greet the user' (adrenalin)
[action] Hi! How can I help you?
[HB #2] hibernate 10s
```

Each heartbeat prints three lines: `stimuli=K (thinking...)` →
`decision: ...` → `hibernate Ns`. That way, even when Self chooses
"No action", you can see at a glance that the program is alive.

> **Note**: actual quality depends on the LLM you connect. Small
> models may not strictly follow the `[THINKING]` / `[DECISION]`
> format — the parser has a fallback (with no markers, the whole
> response is treated as THINKING+DECISION), but strong
> instruction-following models (Qwen 2.5+, Claude, GPT-4-class)
> deliver a much better experience.

---

## Update / repair / uninstall

```bash
krakey update      # pull origin's latest release tag (vX.Y.Z) and reinstall
krakey repair      # force-checkout the current version's release tag (discards local uncommitted changes; asks first)
krakey uninstall   # pip uninstall krakey (keeps repo / config / workspace)
krakey uninstall --full   # also delete the entire repo dir (config + workspace + .venv all gone; asks first)
```

`update` requires a clean working tree (no uncommitted changes); it
will ask you to commit/stash first otherwise.
`repair` is for restoring the repo files to a known release version
after they have been corrupted.
The version number is `[project] version` in `pyproject.toml`; the
git tag `vX.Y.Z` must match.

---

## Phase 0 acceptance criteria

| # | Check | How to verify |
|---|-------|---------------|
| 1 | The program starts and the heartbeat loop logs | `krakey run` prints `[HB #N] ...` lines |
| 2 | Terminal input `hello` → wake → Self → Hypothalamus → Action reply | type `hello` and press Enter; within seconds you see `[action] <reply>` |
| 3 | When idle, the program sleeps for `default_interval` | with no input, observe `hibernate Ns` waiting for the configured seconds |
| 4 | `Ctrl+C` exits | press Ctrl+C; the program exits cleanly |

## How to confirm memory is working

### Live (terminal)

Each heartbeat prints a GM status line:
```
[HB #5] gm: nodes=3 (+1), edges=2 (+2), fatigue=2%
```
- `(+N)` is the per-heartbeat delta. Always `(+0)` means
  `auto_ingest` / `explicit_write` / `compact` never fire.
- Node count = total memory entries. fatigue% =
  `node_count / soft_limit × 100`.

### Offline (SQLite)

The GM database lives at `workspace/data/graph_memory.sqlite` by
default. You can query it with the `sqlite3` CLI at any time (safe
even while Krakey is running — SQLite WAL handles concurrent reads):

```bash
# node count
sqlite3 workspace/data/graph_memory.sqlite "SELECT COUNT(*) FROM gm_nodes"

# 20 most recent nodes (newest first)
sqlite3 workspace/data/graph_memory.sqlite \
  "SELECT id, category, source_type, name FROM gm_nodes ORDER BY id DESC LIMIT 20"

# group by category
sqlite3 workspace/data/graph_memory.sqlite \
  "SELECT category, COUNT(*) FROM gm_nodes GROUP BY category"

# look at edges
sqlite3 workspace/data/graph_memory.sqlite \
  "SELECT na.name, e.predicate, nb.name FROM gm_edges e
   JOIN gm_nodes na ON na.id=e.node_a
   JOIN gm_nodes nb ON nb.id=e.node_b LIMIT 20"

# which auto nodes have been classified asynchronously
sqlite3 workspace/data/graph_memory.sqlite \
  "SELECT id, category, name, json_extract(metadata,'\$.classified') AS classified
   FROM gm_nodes WHERE source_type='auto' ORDER BY id DESC LIMIT 20"
```

Meaning of `source_type`: `auto` = written by `auto_ingest`, not yet
LLM-classified; `explicit` = `explicit_write` (Self said "remember
…"); `compact` = compressed out of the sliding window.

### Confirming the pipeline is alive

After a healthy conversation, you should observe:
1. The terminal's `gm: nodes=N (+1)` increment shows up → tool_feedback is being auto-ingested.
2. `auto` nodes in SQLite gradually flip to `classified=1` → async classification is running.
3. After a long conversation, `compact` nodes appear → the sliding window has overflowed and was compressed.
4. `explicit` nodes appear → Self has said "remember…".

If item 1 stays at `+0`: tools are not feeding back, or the embedder
is erroring. Look for `[runtime] auto_ingest error:` in the terminal.


## Running tests

```bash
pip install -e ".[dev]"
pytest -q
```

All unit + integration tests use a **mock LLM**; no real network
requests are made, so tests pass even in environments without an LLM
service.

```bash
# run a single module
pytest tests/test_hypothalamus.py -q

# verbose output
pytest -v
```

---

## Project layout (brief)

```
KrakeyBot/
├── pyproject.toml          # install / deps / krakey entry-point config
├── config.yaml             # runtime configuration (created by onboard)
├── src/
│   ├── cli/                # `krakey` command line (run/start/stop/onboard/update/...)
│   ├── main.py             # Runtime + main loop
│   ├── self_agent.py       # Self output parser
│   ├── hypothalamus.py     # decision → structured translation
│   ├── llm/client.py       # unified LLM client
│   ├── models/             # config / stimulus / self_model
│   ├── prompt/             # DNA + builder
│   ├── runtime/            # stimulus_buffer / hibernate / fatigue
│   ├── memory/             # GraphMemory / KnowledgeBase / recall
│   ├── sleep/              # 7-phase sleep pipeline
│   ├── dashboard/          # FastAPI + WS + web chat history
│   ├── sandbox/            # SubprocessRunner + guest VM backend
│   ├── interfaces/         # Tool / Channel ABC + Registry
│   └── plugins/
│       ├── loader.py       # plugin discovery + safe import
│       └── builtin/        # built-in plugin projects (search / coding / ...)
├── tests/                  # pytest, all-mock, no network deps
└── workspace/              # runtime data (gitignored)
    ├── data/
    ├── logs/               # daemon.log lives here
    ├── .krakey.pid         # daemon-mode pidfile
    └── plugins/            # user-defined plugins (optional)
```

## FAQ

**Q: I started Krakey and nothing happens.**
- Check that `base_url` in `config.yaml` points to an LLM service that is **actually running**.
- Does `curl http://localhost:8080/v1/models` return anything?
- Maybe Krakey is waiting on hibernate (Self may have chosen a 30s+ interval) — type something and press Enter.

**Q: Self does not reply to the user.**
- In Phase 0 there is no memory, so the context is empty every heartbeat.
- A model that is too small may emit garbled formatting; try a slightly bigger model (Qwen 2.5 7B+ / Claude Haiku+).
- Print the `[DECISION]` content to see what Self actually decides to do.

**Q: `krakey start` fails but `krakey run` works.**
- Check `workspace/logs/daemon.log` — in background mode, both stdout and stderr are redirected there.
- Check whether the pidfile is stale: `krakey status` cleans up invalid pidfiles automatically.

---

## License
MIT
