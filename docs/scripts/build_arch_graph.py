"""Build the interactive architecture graph (library module).

Walks ``krakey/``, parses every ``.py``, and produces

* ``build_graph()`` — a Cytoscape.js-shaped ``{nodes, edges}`` dict
  describing the folder → file → class → method tree plus import +
  reference edges between them.
* ``_render_html(graph)`` — wraps that payload into a self-contained
  HTML page that renders it as a draggable, expandable graph.

Used by the live-reload server (``serve_arch_graph.py``); not run
directly. The server rebuilds the graph in-memory on every ``krakey/``
change and pushes updates to the browser over Server-Sent Events.

Edge resolution is heuristic — we resolve names that come through
``import`` / ``from … import`` statements and ``self.method`` calls
within the same class. Runtime / dynamic dispatch is not tracked.
"""
from __future__ import annotations

import ast
import datetime as _dt
import html
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ROOT / "krakey"

SKIP_DIRS = {"__pycache__", ".pytest_cache"}


# ---------------- module map ----------------


def _module_name_for(path: Path) -> str:
    """``F:/.../krakey/foo/bar.py`` → ``krakey.foo.bar``.

    ``__init__.py`` collapses to the package name (``krakey.foo``).
    """
    rel = path.relative_to(ROOT).with_suffix("")
    parts = list(rel.parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _build_module_map() -> dict[str, Path]:
    out: dict[str, Path] = {}
    for p in SRC.rglob("*.py"):
        if any(s in p.parts for s in SKIP_DIRS):
            continue
        out[_module_name_for(p)] = p
    return out


# ---------------- per-file extraction ----------------


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


def _resolve_relative(
    current_module: str,
    level: int,
    base: str | None,
    is_package: bool,
) -> str:
    """Resolve a relative import like ``from . import x`` (level=1).

    ``is_package`` is true when the current file is ``__init__.py`` — in
    that case the module *is* the package, so ``from . import x``
    resolves to ``current_module.x`` (one fewer level of trim).
    """
    parts = current_module.split(".")
    trim = level - 1 if is_package else level
    if trim >= len(parts):
        pkg_parts: list[str] = []
    elif trim <= 0:
        pkg_parts = list(parts)
    else:
        pkg_parts = parts[:-trim]
    if base:
        pkg_parts = pkg_parts + base.split(".")
    return ".".join(pkg_parts)


def _collect_imports(
    tree: ast.AST,
    current_module: str,
    modmap: dict[str, Path],
    is_package: bool,
) -> dict[str, tuple[str, str | None]]:
    """Map ``name_in_scope -> (target_module, target_qualname_or_None)``.

    ``target_qualname`` is ``None`` when the bound name *is* a module
    (``import X`` or ``from pkg import submodule``).
    """
    bound: dict[str, tuple[str, str | None]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                bind = alias.asname or alias.name.split(".")[0]
                # ``import x.y.z`` binds ``x`` to module ``x.y.z`` (well, ``x``).
                # We treat the bind as the head module unless aliased.
                target = alias.name if alias.asname else alias.name.split(".")[0]
                bound[bind] = (target, None)
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                full = _resolve_relative(
                    current_module, node.level, node.module, is_package
                )
            else:
                full = node.module or ""
            for alias in node.names:
                if alias.name == "*":
                    continue
                bind = alias.asname or alias.name
                sub = f"{full}.{alias.name}" if full else alias.name
                if sub in modmap:
                    bound[bind] = (sub, None)
                else:
                    bound[bind] = (full, alias.name)
    return bound


def _parse_file(path: Path, current_module: str) -> dict:
    raw = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(raw)
    except SyntaxError as e:
        return {
            "module": current_module,
            "doc": f"(parse error: {e})",
            "classes": [],
            "functions": [],
            "tree": None,
            "raw_imports": [],
        }

    classes: list[dict] = []
    functions: list[dict] = []

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            methods = []
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    methods.append(
                        {
                            "name": item.name,
                            "signature": _signature(item),
                            "doc": ast.get_docstring(item) or "",
                            "lineno": item.lineno,
                            "node": item,
                        }
                    )
            classes.append(
                {
                    "name": node.name,
                    "doc": ast.get_docstring(node) or "",
                    "lineno": node.lineno,
                    "methods": methods,
                    "node": node,
                }
            )
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append(
                {
                    "name": node.name,
                    "signature": _signature(node),
                    "doc": ast.get_docstring(node) or "",
                    "lineno": node.lineno,
                    "node": node,
                }
            )

    return {
        "module": current_module,
        "doc": ast.get_docstring(tree) or "",
        "classes": classes,
        "functions": functions,
        "tree": tree,
    }


# ---------------- reference extraction ----------------


def _extract_calls(
    fn_node: ast.AST,
    current_module: str,
    current_class: str | None,
    same_class_methods: set[str],
    file_imports: dict[str, tuple[str, str | None]],
    same_module_classes: set[str],
    same_module_functions: set[str],
) -> set[tuple[str, str]]:
    """Best-effort extraction of (target_module, target_qualname) refs.

    Looks at:

    * ``foo()`` where ``foo`` is in ``file_imports`` or defined in this
      module.
    * ``M.f()`` where ``M`` was imported as a module.
    * ``ClsName.method()`` where ``ClsName`` was imported as a class.
    * ``self.method()`` inside a class method, resolved to the same class.

    Edges to symbols that aren't in the project are dropped at the
    aggregation step.
    """
    targets: set[tuple[str, str]] = set()

    def resolve_name(n: str) -> tuple[str, str] | None:
        if n in file_imports:
            mod, qn = file_imports[n]
            return (mod, qn) if qn is not None else (mod, "")
        if n in same_module_classes or n in same_module_functions:
            return (current_module, n)
        return None

    for n in ast.walk(fn_node):
        if isinstance(n, ast.Call):
            f = n.func
            if isinstance(f, ast.Name):
                t = resolve_name(f.id)
                if t is not None:
                    targets.add(t)
            elif isinstance(f, ast.Attribute):
                v = f.value
                attr = f.attr
                if isinstance(v, ast.Name):
                    if v.id == "self" and current_class:
                        if attr in same_class_methods:
                            targets.add((current_module, f"{current_class}.{attr}"))
                    elif v.id in file_imports:
                        mod, qn = file_imports[v.id]
                        if qn is None:
                            targets.add((mod, attr))
                        else:
                            targets.add((mod, f"{qn}.{attr}"))
        elif isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
            # Reference to a class/function name without calling it
            # (e.g. passing as callback, isinstance, decorator). Useful
            # for relations.
            t = resolve_name(n.id)
            if t is not None:
                targets.add(t)
    return targets


# ---------------- ID helpers ----------------


def _file_id(rel: Path) -> str:
    return rel.as_posix()


def _dir_id(rel_dir: Path) -> str:
    return rel_dir.as_posix() if rel_dir.parts else "krakey"


def _class_id(file_id: str, name: str) -> str:
    return f"{file_id}::{name}"


def _method_id(file_id: str, cls: str, name: str) -> str:
    return f"{file_id}::{cls}.{name}"


def _func_id(file_id: str, name: str) -> str:
    return f"{file_id}::{name}"


# ---------------- main build ----------------


def build_graph() -> dict:
    modmap = _build_module_map()
    # Inverse: file path → module name
    path_to_module = {p: m for m, p in modmap.items()}

    parsed: dict[Path, dict] = {}
    for p in SRC.rglob("*.py"):
        if any(s in p.parts for s in SKIP_DIRS):
            continue
        parsed[p] = _parse_file(p, _module_name_for(p))

    # ---- nodes
    nodes: list[dict] = []
    seen_nodes: set[str] = set()

    def add_node(data: dict) -> None:
        if data["id"] in seen_nodes:
            return
        seen_nodes.add(data["id"])
        nodes.append({"data": data})

    # Folders
    folder_set: set[Path] = set()
    for p in parsed:
        rel = p.relative_to(ROOT)
        d = rel.parent
        while True:
            folder_set.add(d)
            if d == Path("krakey") or d == Path():
                break
            d = d.parent
    folder_set.add(Path("krakey"))

    for d in sorted(folder_set, key=lambda x: (len(x.parts), x.as_posix())):
        if d == Path():
            continue
        parent = d.parent
        parent_id = _dir_id(parent) if parent != Path() else None
        # `krakey` itself has no parent
        if d == Path("krakey"):
            parent_id = None
        add_node(
            {
                "id": _dir_id(d),
                "label": d.name + "/",
                "kind": "dir",
                "parent": parent_id,
                "fullPath": d.as_posix(),
            }
        )

    # Files + their inner classes/methods/functions
    file_id_for_path: dict[Path, str] = {}
    for p, info in parsed.items():
        rel = p.relative_to(ROOT)
        fid = _file_id(rel)
        file_id_for_path[p] = fid
        parent = _dir_id(rel.parent)
        add_node(
            {
                "id": fid,
                "label": rel.name,
                "kind": "file",
                "parent": parent,
                "fullPath": rel.as_posix(),
                "module": info["module"],
                "doc": info["doc"],
                "lineno": 1,
            }
        )

        for c in info["classes"]:
            cid = _class_id(fid, c["name"])
            add_node(
                {
                    "id": cid,
                    "label": c["name"],
                    "kind": "class",
                    "parent": fid,
                    "doc": c["doc"],
                    "lineno": c["lineno"],
                }
            )
            for m in c["methods"]:
                mid = _method_id(fid, c["name"], m["name"])
                add_node(
                    {
                        "id": mid,
                        "label": m["name"],
                        "kind": "method",
                        "parent": cid,
                        "doc": m["doc"],
                        "signature": m["signature"],
                        "lineno": m["lineno"],
                    }
                )
        for f in info["functions"]:
            funid = _func_id(fid, f["name"])
            add_node(
                {
                    "id": funid,
                    "label": f["name"],
                    "kind": "function",
                    "parent": fid,
                    "doc": f["doc"],
                    "signature": f["signature"],
                    "lineno": f["lineno"],
                }
            )

    # ---- edges
    # Map (target_module, target_qualname_or_empty) → graph node id
    target_to_id: dict[tuple[str, str], str] = {}
    for p, info in parsed.items():
        fid = file_id_for_path[p]
        m = info["module"]
        target_to_id[(m, "")] = fid  # whole module → file
        for c in info["classes"]:
            target_to_id[(m, c["name"])] = _class_id(fid, c["name"])
            for meth in c["methods"]:
                target_to_id[(m, f"{c['name']}.{meth['name']}")] = _method_id(
                    fid, c["name"], meth["name"]
                )
        for f in info["functions"]:
            target_to_id[(m, f["name"])] = _func_id(fid, f["name"])

    edges: list[dict] = []
    edge_seen: set[tuple[str, str, str]] = set()

    def add_edge(src: str, tgt: str, kind: str) -> None:
        if src == tgt:
            return
        key = (src, tgt, kind)
        if key in edge_seen:
            return
        edge_seen.add(key)
        edges.append(
            {
                "data": {
                    "id": f"e{len(edges)}",
                    "source": src,
                    "target": tgt,
                    "kind": kind,
                }
            }
        )

    # File → file imports
    for p, info in parsed.items():
        if info.get("tree") is None:
            continue
        fid = file_id_for_path[p]
        is_pkg = p.name == "__init__.py"
        imports = _collect_imports(info["tree"], info["module"], modmap, is_pkg)
        # build set of imported (module, qn) pairs
        for _bind, (mod, qn) in imports.items():
            # Whole-module import → edge to file (or its package __init__)
            if (mod, "") in target_to_id and mod != info["module"]:
                add_edge(fid, target_to_id[(mod, "")], "import")
            # `from X import Y` where Y is a project symbol → edge to that symbol's file too
            if qn is not None:
                key = (mod, qn)
                if key in target_to_id:
                    target_node = target_to_id[key]
                    target_file = target_node.split("::")[0]
                    if target_file != fid:
                        add_edge(fid, target_file, "import")

    # Class/function/method → references
    for p, info in parsed.items():
        if info.get("tree") is None:
            continue
        fid = file_id_for_path[p]
        m = info["module"]
        is_pkg = p.name == "__init__.py"
        imports = _collect_imports(info["tree"], m, modmap, is_pkg)
        same_module_classes = {c["name"] for c in info["classes"]}
        same_module_functions = {f["name"] for f in info["functions"]}

        for c in info["classes"]:
            cls_methods = {meth["name"] for meth in c["methods"]}
            for meth in c["methods"]:
                src_id = _method_id(fid, c["name"], meth["name"])
                refs = _extract_calls(
                    meth["node"],
                    m,
                    c["name"],
                    cls_methods,
                    imports,
                    same_module_classes,
                    same_module_functions,
                )
                for tmod, tqn in refs:
                    target_id = target_to_id.get((tmod, tqn))
                    if target_id is None:
                        # try just the head (class) when we have ClassName.method
                        if "." in tqn:
                            target_id = target_to_id.get((tmod, tqn.split(".")[0]))
                    if target_id is not None and target_id != src_id:
                        add_edge(src_id, target_id, "ref")

        for f in info["functions"]:
            src_id = _func_id(fid, f["name"])
            refs = _extract_calls(
                f["node"],
                m,
                None,
                set(),
                imports,
                same_module_classes,
                same_module_functions,
            )
            for tmod, tqn in refs:
                target_id = target_to_id.get((tmod, tqn))
                if target_id is None and "." in tqn:
                    target_id = target_to_id.get((tmod, tqn.split(".")[0]))
                if target_id is not None and target_id != src_id:
                    add_edge(src_id, target_id, "ref")

    return {"nodes": nodes, "edges": edges}


# ---------------- HTML rendering ----------------


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>KrakeyBot — Architecture Graph</title>
<style>
  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; height: 100%; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; color: #0f172a; background: #f8fafc; }
  #app { display: flex; flex-direction: column; height: 100vh; }
  header { padding: 10px 16px; background: #fff; border-bottom: 1px solid #e2e8f0; display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
  header h1 { font-size: 1.1rem; margin: 0; color: #1e293b; }
  header .subtitle { color: #64748b; font-size: 0.82rem; margin-right: auto; }
  header button, header label { font-size: 0.84rem; background: #fff; border: 1px solid #cbd5e1; color: #334155; padding: 4px 10px; border-radius: 4px; cursor: pointer; }
  header button:hover { background: #f1f5f9; }
  header label { display: inline-flex; gap: 4px; align-items: center; cursor: default; }
  header label input[type="search"] { border: none; outline: none; font-size: 0.84rem; min-width: 160px; }
  #cy { flex: 1; background: linear-gradient(180deg, #fafafa, #f1f5f9); }
  #legend { position: absolute; right: 12px; bottom: 12px; background: rgba(255,255,255,0.92); border: 1px solid #e2e8f0; border-radius: 6px; padding: 8px 10px; font-size: 0.78rem; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }
  #legend .row { display: flex; align-items: center; gap: 6px; margin: 2px 0; }
  #legend .swatch { width: 12px; height: 12px; border-radius: 50%; }
  .swatch.dir { background: #3b82f6; }
  .swatch.file { background: #f59e0b; }
  .swatch.class { background: #a855f7; }
  .swatch.function { background: #22c55e; }
  .swatch.method { background: #ef4444; }
  #tooltip {
    position: absolute;
    pointer-events: none;
    background: #0f172a;
    color: #f8fafc;
    padding: 8px 12px;
    border-radius: 6px;
    font-size: 0.82rem;
    line-height: 1.4;
    max-width: 380px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.25);
    white-space: pre-wrap;
    opacity: 0;
    transition: opacity 0.08s;
    z-index: 50;
  }
  #tooltip.visible { opacity: 1; }
  #tooltip .ttl { font-weight: 600; color: #fbbf24; }
  #tooltip .sig { font-family: "SF Mono", Menlo, Consolas, monospace; color: #93c5fd; font-size: 0.78rem; margin-top: 2px; word-break: break-all; }
  #tooltip .doc { margin-top: 6px; color: #e2e8f0; }
  #tooltip .meta { color: #94a3b8; font-size: 0.74rem; margin-top: 4px; }
  #panel {
    position: absolute;
    left: 12px;
    top: 60px;
    width: 280px;
    max-height: calc(100% - 80px);
    overflow: auto;
    background: rgba(255,255,255,0.97);
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 10px 12px;
    font-size: 0.84rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    display: none;
  }
  #panel.visible { display: block; }
  #panel h3 { margin: 0 0 4px 0; font-size: 0.92rem; color: #1e293b; }
  #panel .kind { color: #64748b; font-size: 0.74rem; text-transform: uppercase; letter-spacing: 0.05em; }
  #panel .sig { font-family: "SF Mono", Menlo, Consolas, monospace; color: #475569; font-size: 0.78rem; margin: 4px 0; word-break: break-all; }
  #panel .doc { white-space: pre-wrap; color: #334155; margin-top: 8px; line-height: 1.5; }
  #panel .meta { color: #94a3b8; font-size: 0.74rem; margin-top: 6px; }
  #panel .close { float: right; cursor: pointer; color: #94a3b8; font-size: 1rem; line-height: 1; padding: 0 0 4px 6px; }
  #panel .close:hover { color: #ef4444; }
  .hint { color: #64748b; font-size: 0.78rem; }
  #live-status {
    font-size: 0.78rem;
    color: #94a3b8;
    white-space: nowrap;
    padding: 2px 8px;
    border: 1px solid #e2e8f0;
    border-radius: 999px;
    background: #fff;
  }
  #live-status.live { color: #16a34a; border-color: #bbf7d0; background: #f0fdf4; }
  #live-status.busy { color: #b45309; border-color: #fde68a; background: #fffbeb; }
  #live-status.dead { color: #dc2626; border-color: #fecaca; background: #fef2f2; }
</style>
<script src="https://unpkg.com/cytoscape@3.30.2/dist/cytoscape.min.js"></script>
<script src="https://unpkg.com/layout-base@2.0.1/layout-base.js"></script>
<script src="https://unpkg.com/cose-base@2.2.0/cose-base.js"></script>
<script src="https://unpkg.com/cytoscape-fcose@2.2.0/cytoscape-fcose.js"></script>
<script src="https://unpkg.com/cytoscape-expand-collapse@4.1.1/cytoscape-expand-collapse.js"></script>
</head>
<body>
<div id="app">
  <header>
    <h1>KrakeyBot — Architecture Graph</h1>
    <span class="subtitle">__SUBTITLE__</span>
    <span id="live-status" title="Live mode is active when this page is served by scripts/serve_arch_graph.py">○ static</span>
    <label><input id="search" type="search" placeholder="Search files / classes / methods…"></label>
    <button id="btn-collapse-all">Collapse all</button>
    <button id="btn-expand-files">Expand to files</button>
    <button id="btn-expand-all">Expand all</button>
    <button id="btn-relayout">Re-layout</button>
    <button id="btn-fit">Fit</button>
  </header>
  <div id="cy"></div>
  <div id="legend">
    <div class="row"><span class="swatch dir"></span> folder</div>
    <div class="row"><span class="swatch file"></span> file</div>
    <div class="row"><span class="swatch class"></span> class</div>
    <div class="row"><span class="swatch function"></span> function</div>
    <div class="row"><span class="swatch method"></span> method</div>
    <div class="row hint">double-click = expand/collapse · click = inspect · right-click = hide/show · drag = move</div>
  </div>
  <div id="panel"><span class="close">×</span><h3 id="panel-title"></h3><div class="kind" id="panel-kind"></div><div class="sig" id="panel-sig"></div><div class="doc" id="panel-doc"></div><div class="meta" id="panel-meta"></div></div>
  <div id="tooltip"></div>
</div>

<script>
const GRAPH = __GRAPH_JSON__;

cytoscape.use(cytoscapeFcose);
cytoscape.use(cytoscapeExpandCollapse);

const cy = cytoscape({
  container: document.getElementById("cy"),
  elements: { nodes: GRAPH.nodes, edges: GRAPH.edges },
  wheelSensitivity: 0.25,
  style: [
    {
      selector: "node",
      style: {
        "label": "data(label)",
        "font-size": 11,
        "color": "#0f172a",
        "text-valign": "center",
        "text-halign": "center",
        "background-color": "#cbd5e1",
        "border-width": 1,
        "border-color": "#94a3b8",
        "min-zoomed-font-size": 5,
      }
    },
    {
      selector: 'node[kind = "dir"]',
      style: {
        "background-color": "#3b82f6",
        "background-opacity": 0.14,
        "border-color": "#3b82f6",
        "border-width": 2,
        "shape": "round-rectangle",
        "padding": 24,
        "font-size": 14,
        "font-weight": 700,
        "color": "#1e3a8a",
        "text-valign": "top",
        "text-halign": "center",
        "text-margin-y": -12,
        "text-background-color": "#dbeafe",
        "text-background-opacity": 0.97,
        "text-background-padding": 5,
        "text-background-shape": "round-rectangle",
        "text-border-width": 1,
        "text-border-color": "#3b82f6",
        "text-border-opacity": 1,
        "corner-radius": 14,
      }
    },
    {
      selector: 'node[kind = "file"]',
      style: {
        "background-color": "#f59e0b",
        "background-opacity": 0.14,
        "border-color": "#f59e0b",
        "border-width": 2,
        "shape": "round-rectangle",
        "padding": 18,
        "font-size": 12,
        "font-weight": 700,
        "color": "#7c2d12",
        "text-valign": "top",
        "text-halign": "center",
        "text-margin-y": -10,
        "text-background-color": "#fef3c7",
        "text-background-opacity": 0.98,
        "text-background-padding": 4,
        "text-background-shape": "round-rectangle",
        "text-border-width": 1,
        "text-border-color": "#f59e0b",
        "text-border-opacity": 1,
      }
    },
    {
      selector: 'node[kind = "class"]',
      style: {
        "background-color": "#a855f7",
        "background-opacity": 0.14,
        "border-color": "#a855f7",
        "border-width": 2,
        "shape": "round-rectangle",
        "padding": 12,
        "font-size": 11,
        "font-weight": 700,
        "color": "#581c87",
        "text-valign": "top",
        "text-halign": "center",
        "text-margin-y": -8,
        "text-background-color": "#f3e8ff",
        "text-background-opacity": 0.98,
        "text-background-padding": 3,
        "text-background-shape": "round-rectangle",
        "text-border-width": 1,
        "text-border-color": "#a855f7",
        "text-border-opacity": 1,
      }
    },
    {
      selector: 'node[kind = "function"]',
      style: {
        "background-color": "#22c55e",
        "border-color": "#15803d",
        "shape": "ellipse",
        "width": 36,
        "height": 36,
        "color": "#0f172a",
      }
    },
    {
      selector: 'node[kind = "method"]',
      style: {
        "background-color": "#ef4444",
        "border-color": "#b91c1c",
        "shape": "ellipse",
        "width": 32,
        "height": 32,
        "color": "#0f172a",
      }
    },
    {
      selector: 'node:selected',
      style: {
        "border-width": 4,
        "border-color": "#fbbf24",
      }
    },
    {
      selector: ".dimmed",
      style: { "opacity": 0.18 }
    },
    {
      selector: ".highlight",
      style: { "border-color": "#fbbf24", "border-width": 4 }
    },
    {
      selector: "edge",
      style: {
        "width": 1.4,
        "line-color": "#94a3b8",
        "curve-style": "bezier",
        "target-arrow-shape": "triangle",
        "target-arrow-color": "#94a3b8",
        "arrow-scale": 0.9,
        "opacity": 0.65,
      }
    },
    {
      selector: 'edge[kind = "import"]',
      style: { "line-color": "#0ea5e9", "target-arrow-color": "#0ea5e9" }
    },
    {
      selector: 'edge[kind = "ref"]',
      style: { "line-color": "#f97316", "target-arrow-color": "#f97316", "line-style": "dashed" }
    },
    {
      selector: "edge.cy-expand-collapse-meta-edge",
      style: {
        "width": 2.4,
        "opacity": 0.9,
        "label": function (ele) {
          const tc = ele.data("totalCount");
          if (typeof tc === "number" && tc > 1) return String(tc);
          const ce = ele.data("collapsedEdges");
          let n = 0;
          if (ce) n = (typeof ce.length === "number") ? ce.length : (ce.size ? ce.size() : 0);
          return n > 1 ? String(n) : "";
        },
        "font-size": 11,
        "font-weight": 700,
        "color": "#0f172a",
        "text-background-color": "#fde68a",
        "text-background-opacity": 1,
        "text-background-padding": 3,
        "text-background-shape": "round-rectangle",
        "text-border-width": 1,
        "text-border-color": "#f59e0b",
        "text-border-opacity": 1,
        "text-rotation": 0,
      }
    },
    {
      selector: "edge.hidden-meta",
      style: { "display": "none" }
    },
    {
      selector: ".edge-highlight",
      style: { "line-color": "#fbbf24", "target-arrow-color": "#fbbf24", "width": 2.6, "opacity": 1 }
    },
    // Right-click hide: the clicked node (and, if it's a compound, every
    // descendant) plus any edge touching the hidden subtree fade to 20%.
    {
      selector: "node.node-hidden",
      style: { "opacity": 0.2 }
    },
    {
      selector: "edge.edge-hidden",
      style: { "opacity": 0.2 }
    },
  ],
  layout: { name: "fcose", animate: false, randomize: true, nodeRepulsion: 9000, idealEdgeLength: 90, padding: 30 },
});

const ec = cy.expandCollapse({
  layoutBy: { name: "fcose", animate: false, randomize: false, nodeRepulsion: 9000, idealEdgeLength: 80 },
  fisheye: false,
  animate: true,
  animationDuration: 220,
  undoable: false,
  cueEnabled: true,
  expandCollapseCueSize: 12,
  // We let the extension produce its native meta-edges first (which
  // may emit one meta-edge per kind, or even per original edge in
  // some versions) and then run our own consolidation pass below to
  // collapse parallel meta-edges between the same source/target pair
  // into a single visible edge labeled with the total count.
  groupEdgesOfSameTypeOnCollapse: true,
});

// ---- Meta-edge consolidation ----
//
// After every collapse/expand the extension may leave several meta-edges
// between the same pair of collapsed compounds (one per kind, or one
// per original edge depending on version). We want a single visible
// edge per pair with a count, so we hide the duplicates and stash the
// total count on the survivor.
function consolidateMetaEdges() {
  const meta = cy.edges(".cy-expand-collapse-meta-edge");
  // Reset prior consolidation marks so re-runs stay correct.
  meta.removeClass("hidden-meta");
  meta.forEach(e => { e.data("totalCount", null); });
  const groups = new Map();
  meta.forEach(e => {
    const key = e.data("source") + "->" + e.data("target");
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(e);
  });
  groups.forEach(edges => {
    if (edges.length <= 1) return;
    let total = 0;
    edges.forEach(e => {
      const ce = e.data("collapsedEdges");
      let n = 0;
      if (ce) n = (typeof ce.length === "number") ? ce.length : (ce.size ? ce.size() : 1);
      total += (n || 1);
    });
    edges[0].data("totalCount", total);
    for (let i = 1; i < edges.length; i++) edges[i].addClass("hidden-meta");
  });
}
cy.on("expandcollapse.aftercollapse", () => { consolidateMetaEdges(); recomputeHidden(); });
cy.on("expandcollapse.afterexpand", () => { consolidateMetaEdges(); recomputeHidden(); });

// ---- Right-click hide (toggle) ----
//
// Right-clicking a node toggles `data.userHidden` on it. The visual
// `.node-hidden` / `.edge-hidden` classes are then recomputed so that:
//   * a node is faded if it OR any ancestor has `userHidden` set
//     (so right-clicking a folder dims the whole subtree),
//   * an edge is faded if either endpoint is faded (covering both
//     fine-grained edges and meta-edges between collapsed compounds).
function recomputeHidden() {
  cy.batch(() => {
    cy.nodes().forEach(n => {
      let hidden = !!n.data("userHidden");
      if (!hidden) {
        n.ancestors().forEach(a => { if (a.data("userHidden")) hidden = true; });
      }
      if (hidden) n.addClass("node-hidden");
      else n.removeClass("node-hidden");
    });
    cy.edges().forEach(e => {
      const s = e.source();
      const t = e.target();
      if ((s && s.hasClass("node-hidden")) || (t && t.hasClass("node-hidden"))) {
        e.addClass("edge-hidden");
      } else {
        e.removeClass("edge-hidden");
      }
    });
  });
}

// Suppress the browser's native context menu on the graph canvas so
// the right-click event reaches Cytoscape cleanly.
document.getElementById("cy").addEventListener("contextmenu", e => e.preventDefault());

cy.on("cxttap", "node", evt => {
  const node = evt.target;
  node.data("userHidden", !node.data("userHidden"));
  recomputeHidden();
});

// Initially collapse everything except the top-level src node.
function collapseAllExceptRoot() {
  const compounds = cy.nodes().filter(n => n.isParent());
  ec.collapse(compounds);
}

function expandToFiles() {
  // Expand only directory compounds
  const dirs = cy.nodes('[kind = "dir"]');
  ec.expandRecursively(dirs);
  // Then collapse files
  const files = cy.nodes('[kind = "file"]');
  ec.collapse(files);
}

function expandAll() {
  ec.expandAll();
}

setTimeout(() => {
  collapseAllExceptRoot();
  // Re-expand the top-level `krakey` container so the user lands on the
  // folder map rather than a single collapsed circle.
  const root = cy.nodes('[kind = "dir"]').filter(n => !n.data("parent"));
  if (root && root.length) ec.expand(root);
  consolidateMetaEdges();
  recomputeHidden();
  cy.fit(undefined, 40);
}, 50);

// ---- Tooltip ----
const tooltipEl = document.getElementById("tooltip");
function showTip(node, evt) {
  const d = node.data();
  let body = "";
  if (d.kind === "dir") {
    body = `<div class="ttl">${escapeHtml(d.fullPath || d.label)}</div>`
         + `<div class="meta">folder · double-click to expand / collapse</div>`;
  } else {
    const sig = d.signature ? `<div class="sig">${escapeHtml(d.signature)}</div>` : "";
    const doc = d.doc ? `<div class="doc">${escapeHtml(truncate(d.doc, 360))}</div>` : `<div class="doc"><i style="color:#94a3b8">(no docstring)</i></div>`;
    const subtitle = d.kind === "file" ? d.module : d.kind;
    body = `<div class="ttl">${escapeHtml(d.label)}</div>`
         + `<div class="meta">${escapeHtml(subtitle || "")}${d.lineno ? " · L" + d.lineno : ""}</div>`
         + sig + doc;
  }
  tooltipEl.innerHTML = body;
  tooltipEl.classList.add("visible");
  positionTip(evt);
}
function positionTip(evt) {
  const x = evt.originalEvent ? evt.originalEvent.clientX : evt.clientX;
  const y = evt.originalEvent ? evt.originalEvent.clientY : evt.clientY;
  const tw = tooltipEl.offsetWidth;
  const th = tooltipEl.offsetHeight;
  const vw = window.innerWidth;
  const vh = window.innerHeight;
  let nx = x + 14;
  let ny = y + 14;
  if (nx + tw > vw - 8) nx = x - tw - 14;
  if (ny + th > vh - 8) ny = y - th - 14;
  tooltipEl.style.left = nx + "px";
  tooltipEl.style.top = ny + "px";
}
function hideTip() {
  tooltipEl.classList.remove("visible");
}
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, ch => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;"
  }[ch]));
}
function truncate(s, n) {
  if (!s) return "";
  s = String(s).trim();
  return s.length > n ? s.slice(0, n - 1) + "…" : s;
}

cy.on("mouseover", "node", evt => showTip(evt.target, evt));
cy.on("mousemove", "node", evt => positionTip(evt));
cy.on("mouseout", "node", () => hideTip());
cy.on("pan zoom", () => hideTip());

// ---- Panel ----
const panelEl = document.getElementById("panel");
function showPanel(node) {
  const d = node.data();
  document.getElementById("panel-title").textContent = d.label || "";
  document.getElementById("panel-kind").textContent = (d.kind || "").toUpperCase()
    + (d.module ? " · " + d.module : (d.fullPath ? " · " + d.fullPath : ""));
  document.getElementById("panel-sig").textContent = d.signature || "";
  document.getElementById("panel-doc").textContent = d.doc || "(no docstring)";
  document.getElementById("panel-meta").textContent = d.lineno ? "line " + d.lineno : "";
  panelEl.classList.add("visible");
}
function hidePanel() { panelEl.classList.remove("visible"); }
panelEl.querySelector(".close").addEventListener("click", hidePanel);

// ---- Single tap: inspect (panel + neighborhood highlight) ----
cy.on("tap", "node", evt => {
  const node = evt.target;
  showPanel(node);
  highlightNeighborhood(node);
});

cy.on("tap", evt => {
  if (evt.target === cy) {
    hidePanel();
    clearHighlight();
  }
});

// ---- Double tap: toggle expand/collapse on the tapped compound ----
//
// Only the directly-tapped node toggles — descendants stay in their
// current state, so each double-click reveals/hides exactly one level
// of one folder (or one file).
cy.on("dbltap", "node", evt => {
  const node = evt.target;
  const canExpand = (typeof ec.isExpandable === "function") && ec.isExpandable(node);
  const canCollapse = (typeof ec.isCollapsible === "function") && ec.isCollapsible(node);
  if (canExpand) {
    ec.expand(node);
  } else if (canCollapse) {
    ec.collapse(node);
  }
});

function highlightNeighborhood(node) {
  // Mark the clicked node + its connected edges, but DON'T dim the
  // rest of the graph — that made everything fade on every click.
  cy.elements().removeClass("edge-highlight").removeClass("highlight");
  node.addClass("highlight");
  node.connectedEdges().addClass("edge-highlight");
}
function clearHighlight() {
  cy.elements().removeClass("dimmed").removeClass("edge-highlight").removeClass("highlight");
}

// ---- Header buttons ----
document.getElementById("btn-collapse-all").addEventListener("click", () => {
  collapseAllExceptRoot();
  const root = cy.nodes('[kind = "dir"]').filter(n => !n.data("parent"));
  if (root && root.length) ec.expand(root);
});
document.getElementById("btn-expand-files").addEventListener("click", expandToFiles);
document.getElementById("btn-expand-all").addEventListener("click", expandAll);
document.getElementById("btn-relayout").addEventListener("click", () => {
  cy.layout({ name: "fcose", animate: true, randomize: false, nodeRepulsion: 9000, idealEdgeLength: 90 }).run();
});
document.getElementById("btn-fit").addEventListener("click", () => cy.fit(undefined, 40));

// ---- Search ----
document.getElementById("search").addEventListener("input", evt => {
  const q = evt.target.value.trim().toLowerCase();
  if (!q) { clearHighlight(); return; }
  const matches = cy.nodes().filter(n => {
    const d = n.data();
    return (d.label || "").toLowerCase().includes(q)
        || (d.module || "").toLowerCase().includes(q)
        || (d.fullPath || "").toLowerCase().includes(q);
  });
  cy.elements().addClass("dimmed");
  matches.removeClass("dimmed");
  matches.ancestors().removeClass("dimmed");
  matches.connectedEdges().removeClass("dimmed");
});

// ---- Live updates over Server-Sent Events ----
//
// When this page is served by `scripts/serve_arch_graph.py`, the
// server pushes an `update` event whenever any file under `krakey/`
// changes. We refetch `/graph.json` and swap elements in-place so
// the user keeps their expand/collapse, hide, pan, and zoom state.
// When the page is opened from disk (file://) there is no server,
// so we silently stay in static mode.
(function setupLive() {
  const statusEl = document.getElementById("live-status");
  function setStatus(text, cls) {
    if (!statusEl) return;
    statusEl.textContent = text;
    statusEl.classList.remove("live", "busy", "dead");
    if (cls) statusEl.classList.add(cls);
  }
  if (typeof EventSource === "undefined") return;
  if (location.protocol !== "http:" && location.protocol !== "https:") return;

  async function applyGraphUpdate(newGraph) {
    // Capture user state.
    const expandedIds = new Set();
    cy.nodes().forEach(n => { if (n.isParent()) expandedIds.add(n.id()); });
    const hiddenIds = new Set();
    cy.nodes().forEach(n => { if (n.data("userHidden")) hiddenIds.add(n.id()); });
    const positions = new Map();
    cy.nodes().forEach(n => {
      // Only leaves carry layout-meaningful positions; compounds derive
      // their bounds from children.
      if (!n.isParent()) {
        const p = n.position();
        positions.set(n.id(), { x: p.x, y: p.y });
      }
    });
    const pan = cy.pan();
    const zoom = cy.zoom();

    cy.batch(() => {
      cy.elements().remove();
      cy.add(newGraph.nodes);
      cy.add(newGraph.edges);
    });

    cy.nodes().forEach(n => {
      const p = positions.get(n.id());
      if (p) n.position(p);
      if (hiddenIds.has(n.id())) n.data("userHidden", true);
    });

    // Re-apply collapse: collapse everything, then re-expand previously
    // expanded compounds. New compounds default to collapsed.
    const allCompounds = cy.nodes().filter(n => n.isParent());
    if (allCompounds.length) ec.collapse(allCompounds);
    const toExpand = cy.nodes().filter(n => expandedIds.has(n.id()));
    if (toExpand.length) toExpand.forEach(n => ec.expand(n));

    consolidateMetaEdges();
    recomputeHidden();
    cy.pan(pan);
    cy.zoom(zoom);
  }

  let es = null;
  function connect() {
    setStatus("● connecting…", null);
    es = new EventSource("/events");
    es.addEventListener("open", () => setStatus("● live", "live"));
    es.addEventListener("error", () => {
      setStatus("● disconnected", "dead");
      try { es.close(); } catch (e) {}
      setTimeout(connect, 2000);
    });
    es.addEventListener("update", async () => {
      setStatus("● updating…", "busy");
      try {
        const r = await fetch("/graph.json", { cache: "no-store" });
        const j = await r.json();
        await applyGraphUpdate(j.graph);
        setStatus("● live · v" + j.version, "live");
      } catch (e) {
        console.error("live update failed", e);
        setStatus("● error", "dead");
      }
    });
  }
  connect();
})();
</script>
</body>
</html>
"""


def _render_html(graph: dict) -> str:
    nodes = graph["nodes"]
    edges = graph["edges"]
    n_dirs = sum(1 for n in nodes if n["data"]["kind"] == "dir")
    n_files = sum(1 for n in nodes if n["data"]["kind"] == "file")
    n_classes = sum(1 for n in nodes if n["data"]["kind"] == "class")
    n_funcs = sum(1 for n in nodes if n["data"]["kind"] in ("function", "method"))
    n_edges = len(edges)
    generated = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")

    subtitle = (
        f"{n_dirs} folders · {n_files} files · {n_classes} classes · "
        f"{n_funcs} defs · {n_edges} edges · generated {generated}"
    )

    payload = json.dumps(graph, ensure_ascii=False)
    out = HTML_TEMPLATE.replace("__GRAPH_JSON__", payload)
    out = out.replace("__SUBTITLE__", html.escape(subtitle, quote=False))
    return out


