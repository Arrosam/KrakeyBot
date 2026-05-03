"""Server-side composition of the SPA from per-view fragments.

Each dashboard view lives under `static/views/<name>/` with:
  - view.html : the `<section id=\"tab-<name>\">...</section>` fragment
  - view.css  : view-scoped CSS

The composer assembles:
  * a single `index.html` by inlining each fragment into the shell
    (`index.template.html`) at `<!-- @@view:<name>@@ -->` markers, and
  * a single `style.css` by concatenating shared/*.css then every
    view's view.css (deterministic order: shell tab order).

Both are computed once at import/first-request and cached in memory.
No file is written to disk; no build step is required. Browser cache
is disabled by middleware so development changes show up on reload.

Routes served:
  GET /           -> composed index.html
  GET /style.css  -> composed stylesheet
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response


_STATIC_DIR = Path(__file__).parent / "static"
_VIEWS_DIR = _STATIC_DIR / "views"
_SHARED_DIR = _STATIC_DIR / "shared"
_SHELL_PATH = _STATIC_DIR / "index.template.html"

# Order matters: the shell's nav renders tabs in this order too, and
# styles cascade predictably when loaded in the same sequence.
_VIEW_ORDER = ["thoughts", "chat", "memory", "prompts", "log", "settings"]

# Shared stylesheets in load order. `theme.css` first (defines vars
# consumed by everything else), then `layout.css` (tabs + header +
# panel primitives).
_SHARED_CSS_ORDER = ["theme.css", "layout.css"]


_composed_html: str | None = None
_composed_css: str | None = None


def _fallback_index() -> str:
    return """<!doctype html>
<html lang="zh">
<head><meta charset="utf-8"><title>Krakey Dashboard</title></head>
<body>
  <h1>Krakey Dashboard</h1>
  <p>Static assets not built yet. The API is up at <code>/api/health</code>.</p>
</body>
</html>
"""


def compose_html() -> str:
    """Inline each view fragment into the shell and return the full HTML."""
    global _composed_html
    if _composed_html is not None:
        return _composed_html

    if not _SHELL_PATH.exists():
        _composed_html = _fallback_index()
        return _composed_html

    shell = _SHELL_PATH.read_text(encoding="utf-8")
    for view in _VIEW_ORDER:
        marker = f"<!-- @@view:{view}@@ -->"
        fragment_path = _VIEWS_DIR / view / "view.html"
        fragment = (
            fragment_path.read_text(encoding="utf-8")
            if fragment_path.exists()
            else f"<!-- missing fragment: {view} -->"
        )
        shell = shell.replace(marker, fragment)
    _composed_html = shell
    return shell


def compose_css() -> str:
    """Concatenate shared stylesheets + every view.css in order."""
    global _composed_css
    if _composed_css is not None:
        return _composed_css

    parts: list[str] = []
    for name in _SHARED_CSS_ORDER:
        p = _SHARED_DIR / name
        if p.exists():
            parts.append(f"/* === shared/{name} === */\n"
                          + p.read_text(encoding="utf-8"))
    for view in _VIEW_ORDER:
        p = _VIEWS_DIR / view / "view.css"
        if p.exists():
            parts.append(f"/* === views/{view}/view.css === */\n"
                          + p.read_text(encoding="utf-8"))
    _composed_css = "\n\n".join(parts)
    return _composed_css


def invalidate_cache() -> None:
    """Drop the in-memory cache so the next request re-reads fragments.

    Useful for tests that mutate the static tree and for a future
    `--dev` flag that disables caching entirely.
    """
    global _composed_html, _composed_css
    _composed_html = None
    _composed_css = None


def register(app: FastAPI) -> None:

    @app.get("/", response_class=HTMLResponse)
    async def index():  # noqa: ANN201
        return HTMLResponse(compose_html(), status_code=200)

    @app.get("/style.css")
    async def style():  # noqa: ANN201
        if not _STATIC_DIR.exists():
            raise HTTPException(status_code=404, detail="no static dir")
        return Response(content=compose_css(), media_type="text/css")
