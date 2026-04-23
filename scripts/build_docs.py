"""Generate docs/architecture.html — per-file source documentation.

Walks `src/`, AST-parses every `.py`, and emits a single self-contained
HTML page where every directory / file / class / method is a collapsible
`<details>` section.

Maintenance rule:
    After editing any file under `src/`, re-run this script and commit
    `docs/architecture.html` alongside your changes. The page is
    generated from module + class + function docstrings, so the
    simplest way to keep it useful is to keep docstrings honest.

Usage:
    python scripts/build_docs.py
"""
from __future__ import annotations

import ast
import datetime as _dt
import html
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
OUT = ROOT / "docs" / "architecture.html"

# Files / dirs to skip entirely during the walk.
SKIP_DIRS = {"__pycache__", ".pytest_cache"}
SKIP_SUFFIXES = {".pyc", ".pyo"}
BINARY_EXT = {".png", ".jpg", ".jpeg", ".gif", ".ico", ".woff", ".woff2"}


# ---------------- data extraction ----------------


def _signature(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    try:
        args = ast.unparse(fn.args)
    except Exception:
        args = "..."
    ret = ""
    if fn.returns is not None:
        try:
            ret = f" -> {ast.unparse(fn.returns)}"
        except Exception:
            ret = ""
    prefix = "async def " if isinstance(fn, ast.AsyncFunctionDef) else "def "
    return f"{prefix}{fn.name}({args}){ret}"


def _summarize_py(path: Path) -> dict:
    raw = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(raw)
    except SyntaxError as e:
        return {
            "kind": "py",
            "error": f"parse error: {e}",
            "bytes": len(raw.encode("utf-8")),
            "lines": raw.count("\n") + 1,
        }

    module_doc = ast.get_docstring(tree) or ""
    classes: list[dict] = []
    functions: list[dict] = []

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            methods: list[dict] = []
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    methods.append({
                        "name": item.name,
                        "signature": _signature(item),
                        "doc": ast.get_docstring(item) or "",
                        "lineno": item.lineno,
                    })
            bases = []
            for b in node.bases:
                try:
                    bases.append(ast.unparse(b))
                except Exception:
                    pass
            classes.append({
                "name": node.name,
                "bases": bases,
                "doc": ast.get_docstring(node) or "",
                "methods": methods,
                "lineno": node.lineno,
            })
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append({
                "name": node.name,
                "signature": _signature(node),
                "doc": ast.get_docstring(node) or "",
                "lineno": node.lineno,
            })

    return {
        "kind": "py",
        "module_doc": module_doc,
        "classes": classes,
        "functions": functions,
        "bytes": len(raw.encode("utf-8")),
        "lines": raw.count("\n") + 1,
    }


def _summarize_text(path: Path) -> dict:
    raw = path.read_text(encoding="utf-8", errors="replace")
    # Try to pull a top comment for purpose
    purpose = ""
    stripped = raw.lstrip()
    if stripped.startswith("<!--"):
        end = stripped.find("-->")
        if end != -1:
            purpose = stripped[4:end].strip()
    elif stripped.startswith("/*"):
        end = stripped.find("*/")
        if end != -1:
            purpose = stripped[2:end].strip().lstrip("*").strip()
    elif stripped.startswith("//"):
        purpose = stripped.split("\n", 1)[0][2:].strip()
    return {
        "kind": path.suffix.lstrip(".") or "text",
        "purpose": purpose,
        "bytes": len(raw.encode("utf-8")),
        "lines": raw.count("\n") + 1,
    }


def _summarize_binary(path: Path) -> dict:
    return {
        "kind": path.suffix.lstrip(".") or "bin",
        "bytes": path.stat().st_size,
        "lines": None,
        "binary": True,
    }


def _walk(root: Path):
    """Yield (relative_path, info) for every file under root."""
    for entry in sorted(root.iterdir()):
        if entry.is_dir():
            if entry.name in SKIP_DIRS:
                continue
            yield from _walk(entry)
        else:
            if entry.suffix in SKIP_SUFFIXES:
                continue
            rel = entry.relative_to(ROOT)
            if entry.suffix == ".py":
                info = _summarize_py(entry)
            elif entry.suffix in BINARY_EXT:
                info = _summarize_binary(entry)
            else:
                info = _summarize_text(entry)
            yield rel, info


# ---------------- tree assembly ----------------


def _build_tree(files: list[tuple[Path, dict]]) -> dict:
    """Group files by directory into a nested dict:
        {"_files": [(name, info), ...], "subdir": {...}}
    """
    root: dict = {"_files": []}
    for rel, info in files:
        parts = rel.parts
        node = root
        for part in parts[:-1]:
            node = node.setdefault(part, {"_files": []})
        node["_files"].append((parts[-1], info, rel))
    return root


# ---------------- HTML rendering ----------------


CSS = """
* { box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
  max-width: 1100px;
  margin: 0 auto;
  padding: 24px;
  color: #0f172a;
  line-height: 1.55;
  background: #fafafa;
}
h1 { font-size: 1.6rem; margin: 0 0 4px 0; }
h2 { font-size: 1.15rem; margin: 24px 0 8px 0; color: #334155; }
.subtitle { color: #64748b; margin-bottom: 8px; }
.stats {
  display: inline-block;
  padding: 8px 12px;
  background: #fff;
  border: 1px solid #e2e8f0;
  border-radius: 6px;
  font-size: 0.9rem;
  color: #475569;
  margin: 4px 0 20px 0;
}
.maintenance {
  background: #fef3c7;
  border: 1px solid #fbbf24;
  border-radius: 6px;
  padding: 10px 14px;
  font-size: 0.88rem;
  color: #78350f;
  margin-bottom: 20px;
}
.maintenance code {
  background: rgba(0,0,0,0.08);
  padding: 1px 5px;
  border-radius: 3px;
}
code {
  font-family: "SF Mono", Menlo, Consolas, monospace;
  font-size: 0.88em;
}
details {
  margin: 2px 0;
}
summary {
  cursor: pointer;
  padding: 4px 6px;
  border-radius: 4px;
  user-select: none;
  list-style: none;
}
summary::-webkit-details-marker { display: none; }
summary::before {
  content: "▸";
  display: inline-block;
  width: 14px;
  color: #94a3b8;
  transition: transform 0.1s;
}
details[open] > summary::before {
  transform: rotate(90deg);
}
summary:hover { background: #f1f5f9; }
.dir > summary {
  font-weight: 600;
  color: #1e40af;
  font-size: 0.98rem;
}
.dir > summary code { font-weight: 600; }
.file > summary {
  background: #fff;
  border: 1px solid #e2e8f0;
  margin: 3px 0;
  padding: 6px 8px;
}
.file > summary:hover { background: #f8fafc; }
.class > summary {
  background: #eff6ff;
  border-left: 3px solid #3b82f6;
  padding-left: 8px;
  margin: 3px 0;
}
.func > summary {
  background: #f0fdf4;
  border-left: 3px solid #22c55e;
  padding-left: 8px;
  margin: 3px 0;
}
.method > summary {
  background: #fef2f2;
  border-left: 3px solid #f87171;
  padding-left: 8px;
  margin: 2px 0;
  font-size: 0.92em;
}
.body {
  padding: 6px 10px 6px 22px;
  border-left: 1px dashed #e2e8f0;
  margin-left: 8px;
}
.meta {
  color: #64748b;
  font-size: 0.82em;
  font-weight: normal;
}
.tag {
  display: inline-block;
  padding: 1px 6px;
  border-radius: 3px;
  font-size: 0.72em;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  vertical-align: middle;
  margin-right: 4px;
}
.tag-class { background: #dbeafe; color: #1e40af; }
.tag-func { background: #dcfce7; color: #166534; }
.tag-method { background: #fee2e2; color: #991b1b; }
.tag-async { background: #fef3c7; color: #92400e; }
.tag-empty { background: #f1f5f9; color: #64748b; }
.doc {
  white-space: pre-wrap;
  color: #334155;
  font-size: 0.92em;
  margin: 4px 0;
}
.no-doc { color: #94a3b8; font-style: italic; font-size: 0.88em; }
.signature {
  font-family: "SF Mono", Menlo, Consolas, monospace;
  font-size: 0.82em;
  color: #64748b;
  margin: 2px 0 6px 0;
  word-break: break-all;
}
.error {
  color: #991b1b;
  background: #fef2f2;
  padding: 4px 8px;
  border-radius: 4px;
  font-size: 0.88em;
}
.controls {
  margin: 0 0 16px 0;
  font-size: 0.88rem;
}
.controls button {
  background: #fff;
  border: 1px solid #cbd5e1;
  padding: 4px 10px;
  border-radius: 4px;
  cursor: pointer;
  font-size: 0.88rem;
  margin-right: 6px;
}
.controls button:hover { background: #f1f5f9; }
"""


SCRIPT = """
document.getElementById("expand-all").addEventListener("click", () => {
  document.querySelectorAll("details").forEach(d => d.open = true);
});
document.getElementById("collapse-all").addEventListener("click", () => {
  document.querySelectorAll("details").forEach(d => d.open = false);
});
document.getElementById("expand-dirs").addEventListener("click", () => {
  document.querySelectorAll("details").forEach(d => d.open = false);
  document.querySelectorAll("details.dir").forEach(d => d.open = true);
});
"""


def _fmt_size(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n / 1024 / 1024:.2f} MB"


def _h(s: str) -> str:
    return html.escape(s, quote=False)


def _render_doc(doc: str) -> str:
    if not doc.strip():
        return '<p class="no-doc">(no docstring)</p>'
    return f'<p class="doc">{_h(doc.strip())}</p>'


def _render_method(m: dict, tag_class: str = "tag-method", det_class: str = "method") -> str:
    is_async = m["signature"].startswith("async")
    name_part = _h(m["name"])
    async_tag = ' <span class="tag tag-async">async</span>' if is_async else ""
    lineno = f' <span class="meta">L{m["lineno"]}</span>'
    summary = (
        f'<summary>'
        f'<span class="tag {tag_class}">{tag_class.split("-")[1]}</span>'
        f'<code>{name_part}</code>{async_tag}{lineno}'
        f'</summary>'
    )
    sig = f'<div class="signature">{_h(m["signature"])}</div>'
    doc = _render_doc(m["doc"])
    return f'<details class="{det_class}">{summary}<div class="body">{sig}{doc}</div></details>'


def _render_class(c: dict) -> str:
    bases = ""
    if c["bases"]:
        bases = f' <span class="meta">({_h(", ".join(c["bases"]))})</span>'
    lineno = f' <span class="meta">L{c["lineno"]}</span>'
    summary = (
        f'<summary>'
        f'<span class="tag tag-class">class</span>'
        f'<code>{_h(c["name"])}</code>{bases}{lineno}'
        f'</summary>'
    )
    parts = [_render_doc(c["doc"])]
    if c["methods"]:
        parts.append(f'<p class="meta">{len(c["methods"])} method(s)</p>')
        for m in c["methods"]:
            parts.append(_render_method(m))
    else:
        parts.append('<p class="no-doc">(no methods)</p>')
    return f'<details class="class">{summary}<div class="body">{"".join(parts)}</div></details>'


def _render_py_file(info: dict) -> str:
    if "error" in info:
        return f'<div class="error">{_h(info["error"])}</div>'
    parts = [_render_doc(info["module_doc"])]
    n_cls = len(info["classes"])
    n_fn = len(info["functions"])
    parts.append(
        f'<p class="meta">{n_cls} class(es) · {n_fn} top-level function(s)</p>'
    )
    for c in info["classes"]:
        parts.append(_render_class(c))
    for f in info["functions"]:
        parts.append(_render_method(f, tag_class="tag-func", det_class="func"))
    if not info["classes"] and not info["functions"]:
        parts.append('<p class="no-doc">(module body only — no classes or top-level functions)</p>')
    return "".join(parts)


def _render_text_file(info: dict) -> str:
    p = info.get("purpose") or ""
    body = _render_doc(p)
    return f'<p class="meta">{_h(info["kind"])} · {info["lines"]} line(s)</p>{body}'


def _render_binary_file(info: dict) -> str:
    return f'<p class="meta">binary · {_h(info["kind"])}</p>'


def _render_file(name: str, info: dict, rel: Path) -> str:
    size_str = _fmt_size(info["bytes"])
    if info.get("binary"):
        lines_str = ""
    elif info.get("lines") is not None:
        lines_str = f' · {info["lines"]} lines'
    else:
        lines_str = ""
    meta = f'<span class="meta">{size_str}{lines_str}</span>'
    # tiny purpose teaser in summary
    teaser = ""
    if info.get("kind") == "py":
        doc = info.get("module_doc", "")
        if doc:
            first = doc.split("\n", 1)[0].strip()
            if len(first) > 80:
                first = first[:77] + "..."
            teaser = f' <span class="meta">— {_h(first)}</span>'
    elif info.get("purpose"):
        first = info["purpose"].split("\n", 1)[0].strip()
        if len(first) > 80:
            first = first[:77] + "..."
        teaser = f' <span class="meta">— {_h(first)}</span>'

    summary = f'<summary><code>{_h(name)}</code> {meta}{teaser}</summary>'
    if info.get("kind") == "py":
        body = _render_py_file(info)
    elif info.get("binary"):
        body = _render_binary_file(info)
    else:
        body = _render_text_file(info)
    anchor = rel.as_posix().replace("/", "-")
    return f'<details class="file" id="f-{_h(anchor)}">{summary}<div class="body">{body}</div></details>'


def _render_dir(name: str, node: dict, depth: int = 0) -> str:
    # Count files recursively + classes + functions
    total_files = 0
    total_classes = 0
    total_functions = 0
    total_bytes = 0

    def _count(n: dict):
        nonlocal total_files, total_classes, total_functions, total_bytes
        for _, info, _ in n.get("_files", []):
            total_files += 1
            total_bytes += info.get("bytes", 0)
            if info.get("kind") == "py":
                total_classes += len(info.get("classes") or [])
                total_functions += len(info.get("functions") or [])
                for c in info.get("classes") or []:
                    total_functions += len(c.get("methods") or [])
        for k, v in n.items():
            if k != "_files":
                _count(v)

    _count(node)

    parts = []
    # directories first
    for subname in sorted(k for k in node if k != "_files"):
        parts.append(_render_dir(subname, node[subname], depth + 1))
    # then files
    for fname, info, rel in node.get("_files", []):
        parts.append(_render_file(fname, info, rel))

    meta = (
        f'<span class="meta">{total_files} file(s) · '
        f'{_fmt_size(total_bytes)} · {total_classes} class · '
        f'{total_functions} def</span>'
    )
    summary = f'<summary><code>{_h(name)}/</code> {meta}</summary>'
    # Top-level dir auto-open
    opener = " open" if depth == 0 else ""
    return f'<details class="dir"{opener}>{summary}<div class="body">{"".join(parts)}</div></details>'


# ---------------- entry ----------------


def main() -> None:
    files = list(_walk(SRC))
    tree = _build_tree(files)

    # stats
    total_files = len(files)
    total_lines = sum((i.get("lines") or 0) for _, i in files)
    total_bytes = sum(i.get("bytes", 0) for _, i in files)
    total_classes = 0
    total_functions = 0
    for _, info in files:
        if info.get("kind") == "py":
            total_classes += len(info.get("classes") or [])
            total_functions += len(info.get("functions") or [])
            for c in info.get("classes") or []:
                total_functions += len(c.get("methods") or [])

    generated = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")

    body_parts = [_render_dir("src", tree.get("src", {"_files": []}), depth=0)]

    html_out = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>KrakeyBot — Source Documentation</title>
<style>{CSS}</style>
</head>
<body>
<h1>KrakeyBot — Source Documentation</h1>
<p class="subtitle">Per-file breakdown of every module under <code>src/</code>.
Every directory, file, class, and method below is a collapsible section.</p>
<div class="stats">
  {total_files} files · {_fmt_size(total_bytes)} · {total_lines:,} lines ·
  {total_classes} classes · {total_functions} defs
</div>
<div class="maintenance">
  <strong>Maintenance:</strong> regenerate with
  <code>python scripts/build_docs.py</code> after edits under
  <code>src/</code>. Last generated: {generated}.
</div>
<div class="controls">
  <button id="expand-dirs">Directories only</button>
  <button id="expand-all">Expand all</button>
  <button id="collapse-all">Collapse all</button>
</div>
{"".join(body_parts)}
<script>{SCRIPT}</script>
</body>
</html>
"""

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html_out, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")
    print(
        f"  {total_files} files · {_fmt_size(total_bytes)} · "
        f"{total_lines:,} lines · {total_classes} classes · "
        f"{total_functions} defs"
    )


if __name__ == "__main__":
    main()
