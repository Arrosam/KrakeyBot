# workspace/plugins/

Drop **plugin projects** here. One folder per project.

Each project folder contains a module (`<project>/__init__.py` or a
single-file `<project>.py`) that exposes a factory returning zero or
more tools + zero or more sensories. Components in one project
can share state — that's exactly why projects exist (e.g. a Telegram
project ships sensory + reply tool that share one HTTP client).

See [`PLUGINS.md`](../../PLUGINS.md) at the repo root for the full
contract, manifest format, and examples.

Anything in this directory except this README is gitignored.
