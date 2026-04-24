# Project conventions for Claude Code

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
