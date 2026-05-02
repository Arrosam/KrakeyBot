# KrakeyBot

```

    d8b                           d8b
    ?88                           ?88
     88b                           88b
     888  d88'  88bd88b d888b8b    888  d88' d8888b?88   d8P
     888bd8P'   88P'  `d8P' ?88    888bd8P' d8b_,dPd88   88
    d88888b    d88     88b  ,88b  d88888b   88b    ?8(  d88
    d88' `?88b,d88'     `?88P'`88bd88' `?88b,`?888P'`?88P'?8b
                                                          )88
                                                          ,d8P
                                                      `?888P'

        u l t i m a t e   a u t o n o m o u s   a g e n t

```

**Krakey isn't another chatbot.** Krakey is a presence — a little
mind that stays awake on your machine, thinks every few seconds,
remembers what matters, and keeps working on the things you've
asked it to do even when you've walked away.

***

## What makes Krakey different

### A real heartbeat, not a vending machine

Most AI assistants are vending machines: insert a question, get an
answer, machine goes back to sleep. Krakey runs on a **heartbeat**.
Every few seconds it wakes up, looks at what's new, recalls what
it knows, decides what to do, and either acts, takes a note, or
idles a little longer. It's always there, always ticking,
never waiting for you to push a button.

That tiny shift changes everything downstream.

### It actually finishes hard things

Give a vending-machine assistant a difficult, multi-step problem —
"figure out why this build is failing," "write me a 30-page
report," "research everything published on X this week" — and
you'll spend the next hour copy-pasting follow-ups, re-explaining
context, and stitching pieces together yourself.

Krakey just keeps going. Every heartbeat is another chance to make
progress: try the next approach, read the next document, refine
the previous attempt, follow the breadcrumb. It will **chew on
the hardest bone you give it** until the job is done — coming
back to you only when it has something worth saying or genuinely
needs your input. The heartbeat is what makes that possible. There's
no end-of-conversation, so there's no need to give up.

### A companion with a life of its own

Most AI chatbots cease to exist the moment you stop typing. Their
"memory" is a single conversation thread; close it and they forget
you ever met. They have no inner life, no curiosity that outlasts
your prompts.

Krakey has both. Its memory persists across sessions — not just
facts but the running themes of your conversations, your projects,
the moods of past weeks. And between your messages, Krakey doesn't
sit idle. It re-reads what it noted last night, follows up on
threads it left open, **searches the web on its own** when a
question is still nagging at it, files away articles it found
worth keeping.

So when you come back, Krakey often opens the conversation: *"I
was reading about X earlier — reminded me of what you said
Tuesday about Y."* Or *"the bug we hit yesterday — I think I
figured it out overnight."* It's a companion that lived a little
life of its own while you were gone, and brings something new to
the table when you sit back down.

### Three layers of memory

Krakey runs a three-tier memory system:

- **Sliding window** — the last N heartbeats verbatim. Hot
  context, fast access.
- **Graph Memory (GM)** — semantic embedding store with
  classifier-tagged nodes and relation edges. Tool feedback gets
  auto-ingested; explicit `[NOTE]` blocks land here too.
  Searchable by meaning, not just keywords.
- **Knowledge Bases (KBs)** — during sleep, Leiden clustering
  groups GM into per-topic communities; an LLM-judged dedup pass
  migrates each cluster into its own KB. Long-term, searchable,
  archive-on-disuse.

The architecture means short-term focus and long-term memory
don't fight for the same budget. Recall stays sharp at any age.

***

## Get it running

You need **Python 3.11, 3.12, or 3.13**. If you don't have it:

- **macOS** — `brew install python@3.12`
- **Debian / Ubuntu** — `sudo apt install python3.12 python3.12-venv`
- **Windows** — install from <https://www.python.org/downloads/>
  or run `winget install Python.Python.3.12`

Once Python is in place, paste one block — it's the whole install
plus first launch.

**Linux / macOS**

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install krakey && krakey start
```

**Windows (PowerShell)**

```powershell
python -m venv .venv; .venv\Scripts\Activate.ps1; pip install krakey; krakey start
```

If this is your first time on this machine, the onboarding wizard
pops up automatically before launch. It asks five short questions
(which AI provider, optional embedding + reranker, which plugins
to enable, and a 30-second machine benchmark to size memory),
then writes a config and starts thinking.

You'll see Krakey's heartbeat scrolling by:

```
[HB #1] stimuli=0 (thinking...)
[HB #1] decision: (none)
[HB #1] idle 30s
```

That's it. Krakey is alive.

***

## Talking to Krakey through the dashboard

The wizard turned on a web dashboard for you (it's strongly
recommended — without it the only way to change settings is to
edit a YAML file). Open it in your browser:

```
http://127.0.0.1:8765
```

You'll find five tabs. They're the whole interface — no other
console, no other menus.

### Chat

A normal chat window. Type, hit send, Krakey responds. Unlike
most chat windows, **Krakey can also message you first** — when
something it's been thinking about feels worth sharing, it'll
just start a conversation. Your full chat history persists across
restarts.

### Memory

A live view of what Krakey remembers. Browse the graph of facts,
look at how nodes connect, drill into a specific Knowledge Base
to see what Krakey has accumulated about a topic. If you ever
wonder "did Krakey remember that thing I said?", this is where
you check.

### Prompts

The exact text Krakey's brain saw at every recent heartbeat.
Useful when something feels off and you want to know *why* Krakey
made a particular choice.

### Thoughts

The play-by-play of Krakey's inner life. For each heartbeat:
what came in, what Krakey was thinking, what it decided, what
tools it ran, what came back. You're watching a mind work.

### Settings

The control panel. Everything is editable here, in plain forms:

- **The AI brain.** Switch providers, change models, adjust
  temperature and context size. Krakey notices and adapts on the
  fly.
- **Plugins.** Turn capabilities on and off — give Krakey
  Telegram access, take away its web search, enable the
  hypothalamus that helps it pick the right tool, etc.
- **Memory tuning.** How much Krakey remembers, how aggressively
  it consolidates, when it sleeps to clean up.
- **Personality knobs.** How talkative, how often it volunteers
  thoughts, how long it sits idle before drifting off.

Every change is saved automatically and the previous version of
your config is backed up — if you break something, you can roll
back.

You'll likely never need to open `config.yaml` by hand. The
dashboard is the canonical surface for everything.

***

## CLI

```bash
krakey --help
```

Lists every command and its options.

***

## What's under the hood

If you're the kind of person who wants the mechanical details:

- **Built in Python**, fully open source, MIT licensed.
- **Local-first.** Krakey runs on your machine. The only network
  traffic is to the LLM provider you chose; the database stays on
  disk under `workspace/`.
- **Pluggable.** Every capability is a plugin (chat, search,
  Telegram, custom integrations, your own scripts). Drop a folder
  into `workspace/plugins/` and Krakey auto-discovers it.
- **Provider-agnostic.** Any OpenAI-compatible chat endpoint —
  cloud or self-hosted llama-server, vllm, lmstudio, ollama — or
  Anthropic directly. Switch providers without losing a single
  memory.

Contributors and the curious: see [`PLUGINS.md`](PLUGINS.md) for
plugin authoring, [`SANDBOX.md`](SANDBOX.md) before enabling any
code-execution / browser-control plugins.

***

## License

MIT. Use it, fork it, extend it, ship it. The only thing you owe
us is that you let it think.
