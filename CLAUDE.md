# Project conventions for Claude Code

## 🔒 Core architectural invariant — plugins are strictly additive

**Disabling or removing ANY plugin (Reflects, tentacles, sensories)
must NOT break the runtime's core loop.** Set 2026-04-25 by Samuel.

This is load-bearing. Every plugin call site in `src/` must have a
graceful fallback — null-object (e.g. `NoopRecall`), soft-fail (e.g.
tentacle dispatch produces a `Unknown tentacle: X` system event for
Self instead of raising), or a phase skip. **No `raise RuntimeError`
when a plugin is missing.**

Test the invariant: `tests/test_zero_plugin_runtime.py` runs Runtime
with all Reflects unregistered + zero tentacles + zero sensories and
asserts it completes a heartbeat. New code that requires a plugin
to function will break that test — fix the missing-plugin path
before merging.

UX defaults (which Reflects auto-register, what tentacles ship
in-tree) are a separate concern from this invariant. Default-on
plugins exist for convenience; the architecture must work with
ALL of them disabled.

## Source documentation — regenerate after edits

`docs/architecture.html` is an auto-generated per-file / per-class /
per-method reference, built by AST-walking `src/`. After **any** edit
under `src/` (adding a file, renaming a class, changing a docstring,
etc.), regenerate and commit alongside the code change:

```
python scripts/build_docs.py
git add docs/architecture.html
```

If the edit was purely internal to a function body and no
signatures / docstrings / file layout changed, regeneration is still
cheap (~1 second) and harmless — just run it. The HTML only changes
when something user-visible changes, so diffs stay small.

Do not hand-edit `docs/architecture.html`. Edit the corresponding
docstring in `src/` and re-run the generator.

## Commit messages

- No `Co-Authored-By: Claude` trailer on commits.
- Follow the existing commit style in this repo: short imperative
  subject, explanatory body describing the "why".
