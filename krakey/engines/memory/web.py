"""Memory engine self-hosted web service.

Provides a pure factory ``create_memory_app(engine)`` that builds a
FastAPI application over a ``GraphMemoryEngine`` instance.  No port
binding happens here — the app is testable via httpx ASGI transport
and is started by ``ThreadedMemoryWebServer`` in ``web_server.py``.
"""
from __future__ import annotations

import asyncio
from typing import Any

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Helper — trim raw embedding to a bool flag so payloads stay sane
# ---------------------------------------------------------------------------

def _serialize_node(n: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of *n* with the raw ``embedding`` replaced by
    ``has_embedding: bool``.  All other fields are passed through as-is."""
    out = dict(n)
    emb = out.pop("embedding", None)
    out["has_embedding"] = emb is not None and len(emb) > 0
    return out


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class NodeCreateBody(BaseModel):
    name: str
    category: str
    description: str
    importance: float = 1.0


class NodePatchBody(BaseModel):
    category: str | None = None
    description: str | None = None
    importance: float | None = None
    metadata: dict[str, Any] | None = None


class EdgeCreateBody(BaseModel):
    predicate: str
    source_id: int | None = None
    source_name: str | None = None
    target_id: int | None = None
    target_name: str | None = None


class KBCreateBody(BaseModel):
    kb_id: str
    name: str
    description: str = ""
    topics: list[str] | None = None


class KBPatchBody(BaseModel):
    archived: bool | None = None


class KBEntryCreateBody(BaseModel):
    content: str
    tags: list[str] | None = None
    importance: float = 1.0


class SleepBody(BaseModel):
    reason: str = ""


class BenchmarkBody(BaseModel):
    sizes: list[int] | None = None
    dim: int = 384
    repeats: int = 10


# ---------------------------------------------------------------------------
# HTML browser page
# ---------------------------------------------------------------------------

_BROWSER_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Memory Browser</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 0; padding: 16px; background: #f5f5f5; color: #222; }
    h1 { margin-bottom: 4px; font-size: 1.4rem; }
    h2 { font-size: 1.1rem; margin-top: 24px; border-bottom: 1px solid #ccc; padding-bottom: 4px; }
    .stats { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 16px; }
    .stat { background: #fff; border: 1px solid #ddd; border-radius: 6px; padding: 10px 18px; text-align: center; }
    .stat-val { font-size: 1.5rem; font-weight: bold; }
    .stat-lbl { font-size: 0.75rem; color: #666; }
    table { border-collapse: collapse; width: 100%; background: #fff; border-radius: 6px; overflow: hidden; margin-top: 8px; }
    th, td { text-align: left; padding: 8px 10px; font-size: 0.85rem; border-bottom: 1px solid #eee; }
    th { background: #f0f0f0; font-weight: 600; }
    tr:last-child td { border-bottom: none; }
    .badge { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 0.75rem;
             background: #e0e7ff; color: #3730a3; }
    ul.kb-list { list-style: none; padding: 0; margin: 0; }
    ul.kb-list li { background: #fff; border: 1px solid #ddd; border-radius: 6px; padding: 10px 14px;
                    margin-bottom: 8px; font-size: 0.85rem; }
    .err { color: #b91c1c; font-size: 0.85rem; }
    #loading { color: #555; font-style: italic; }
  </style>
</head>
<body>
  <h1>Memory Browser</h1>
  <p id="loading">Loading...</p>

  <div id="stats-section" style="display:none">
    <h2>Stats</h2>
    <div class="stats" id="stats-boxes"></div>
    <div id="by-cat-section"></div>
  </div>

  <div id="nodes-section" style="display:none">
    <h2>Graph Memory Nodes <small id="node-count"></small></h2>
    <table id="nodes-table">
      <thead><tr>
        <th>ID</th><th>Name</th><th>Category</th>
        <th>Description</th><th>Importance</th><th>Emb?</th>
      </tr></thead>
      <tbody id="nodes-body"></tbody>
    </table>
  </div>

  <div id="kbs-section" style="display:none">
    <h2>Knowledge Bases</h2>
    <ul class="kb-list" id="kbs-list"></ul>
  </div>

  <script>
    const esc = s => String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');

    async function load() {
      try {
        const [stats, nodes, kbs] = await Promise.all([
          fetch('/api/gm/stats').then(r => r.json()),
          fetch('/api/gm/nodes?limit=200').then(r => r.json()),
          fetch('/api/kbs').then(r => r.json()),
        ]);

        // Stats
        document.getElementById('stats-boxes').innerHTML = `
          <div class="stat"><div class="stat-val">${stats.total_nodes}</div><div class="stat-lbl">Nodes</div></div>
          <div class="stat"><div class="stat-val">${stats.total_edges}</div><div class="stat-lbl">Edges</div></div>
        `;
        const byCat = stats.by_category || {};
        const catRows = Object.entries(byCat).map(([k,v]) =>
          `<tr><td>${esc(k)}</td><td>${v}</td></tr>`).join('');
        document.getElementById('by-cat-section').innerHTML = catRows
          ? `<table><thead><tr><th>Category</th><th>Count</th></tr></thead><tbody>${catRows}</tbody></table>`
          : '';
        document.getElementById('stats-section').style.display = '';

        // Nodes
        document.getElementById('node-count').textContent = `(${nodes.count} total, showing ${nodes.nodes.length})`;
        const tbody = document.getElementById('nodes-body');
        tbody.innerHTML = nodes.nodes.map(n => `
          <tr>
            <td>${n.id}</td>
            <td>${esc(n.name)}</td>
            <td><span class="badge">${esc(n.category)}</span></td>
            <td>${esc(n.description || '')}</td>
            <td>${typeof n.importance === 'number' ? n.importance.toFixed(2) : n.importance}</td>
            <td>${n.has_embedding ? '&#10003;' : ''}</td>
          </tr>`).join('');
        document.getElementById('nodes-section').style.display = '';

        // KBs
        const kbList = document.getElementById('kbs-list');
        if (!kbs.kbs || kbs.kbs.length === 0) {
          kbList.innerHTML = '<li style="color:#888">No knowledge bases.</li>';
        } else {
          kbList.innerHTML = kbs.kbs.map(kb => `
            <li>
              <strong>${esc(kb.name)}</strong>
              <span style="color:#888; font-size:0.8rem; margin-left:8px">${esc(kb.kb_id)}</span>
              <span style="float:right; color:#666">${kb.entry_count ?? 0} entries</span>
              ${kb.description ? `<div style="color:#555; margin-top:4px">${esc(kb.description)}</div>` : ''}
            </li>`).join('');
        }
        document.getElementById('kbs-section').style.display = '';

        document.getElementById('loading').style.display = 'none';
      } catch (e) {
        document.getElementById('loading').textContent = '';
        document.getElementById('loading').className = 'err';
        document.getElementById('loading').textContent = 'Error loading data: ' + e.message;
      }
    }

    load();
  </script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_memory_app(engine) -> FastAPI:
    """Build the memory engine's browse/edit web app over the given
    ``GraphMemoryEngine``.  Pure factory — no port binding; testable via
    httpx ASGI transport."""

    app = FastAPI(title="Memory Browser", docs_url="/docs", redoc_url=None)

    # ------------------------------------------------------------------
    # Browser UI
    # ------------------------------------------------------------------

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def browser_page():
        return HTMLResponse(content=_BROWSER_HTML)

    # ------------------------------------------------------------------
    # GM read endpoints
    # ------------------------------------------------------------------

    @app.get("/api/gm/nodes")
    async def gm_nodes(
        category: str | None = Query(default=None),
        limit: int = Query(default=200, ge=1),
    ):
        nodes = await engine.list_nodes(category=category, limit=limit)
        serialized = [_serialize_node(n) for n in nodes]
        return {"count": len(serialized), "nodes": serialized}

    @app.get("/api/gm/edges")
    async def gm_edges(limit: int = Query(default=500, ge=1)):
        edges = await engine.list_edges_named(limit=limit)
        return {"count": len(edges), "edges": edges}

    @app.get("/api/gm/stats")
    async def gm_stats():
        total_nodes, total_edges, by_cat, by_src = await asyncio.gather(
            engine.count_nodes(),
            engine.count_edges(),
            engine.counts_by_category(),
            engine.counts_by_source(),
        )
        return {
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "by_category": by_cat,
            "by_source": by_src,
        }

    @app.get("/api/kbs")
    async def list_kbs():
        kbs = await engine.list_kbs(include_archived=False)
        return {"kbs": kbs}

    @app.get("/api/kb/{kb_id}/entries")
    async def kb_entries(
        kb_id: str,
        limit: int = Query(default=200, ge=1, le=2000),
    ):
        try:
            kb = await engine.open_kb(kb_id)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"KB '{kb_id}' not found")
        entries = await kb.list_active_entries(limit=limit)
        return {"kb_id": kb_id, "count": len(entries), "entries": entries}

    # ------------------------------------------------------------------
    # GM edit endpoints
    # ------------------------------------------------------------------

    @app.post("/api/gm/nodes", status_code=201)
    async def create_node(body: NodeCreateBody):
        new_id = await engine.insert_node(
            name=body.name,
            category=body.category,
            description=body.description,
            importance=body.importance,
        )
        return {"id": new_id}

    @app.patch("/api/gm/nodes/{node_id}")
    async def patch_node(node_id: int, body: NodePatchBody):
        current = await engine.get_node(node_id)
        if current is None:
            raise HTTPException(status_code=404, detail=f"Node {node_id} not found")

        if body.category is not None:
            await engine.update_node_category(current["name"], body.category)
            # Refresh so that subsequent upsert uses the updated category
            current = await engine.get_node(node_id) or current

        if body.description is not None or body.importance is not None:
            # Use the explicit-edit primitive (SET), NOT upsert_node — the
            # latter BUMPS importance by +0.5 for an existing node, so it
            # could never set the exact value the operator typed.
            await engine.set_node_fields(
                node_id,
                description=body.description,
                importance=body.importance,
            )

        if body.metadata is not None:
            await engine.set_metadata(node_id, body.metadata)

        return {"ok": True}

    @app.delete("/api/gm/nodes/{node_id}")
    async def delete_node(node_id: int):
        await engine.delete_node(node_id)
        return {"ok": True}

    @app.post("/api/gm/edges", status_code=201)
    async def create_edge(body: EdgeCreateBody):
        # Resolve source
        src: int | None = body.source_id
        if src is None and body.source_name is not None:
            src = await engine.find_by_name(body.source_name)
        if src is None:
            raise HTTPException(
                status_code=400,
                detail="source endpoint unresolved — provide source_id or a valid source_name",
            )

        # Resolve target
        tgt: int | None = body.target_id
        if tgt is None and body.target_name is not None:
            tgt = await engine.find_by_name(body.target_name)
        if tgt is None:
            raise HTTPException(
                status_code=400,
                detail="target endpoint unresolved — provide target_id or a valid target_name",
            )

        # A self-loop (src == tgt) or a cycle-closing edge is a benign
        # "skip", not a server error — the underlying primitive raises
        # ValueError on self-loops, so translate that into the same
        # skipped-shape the cycle-check returns instead of a 500.
        if src == tgt:
            return {"skipped": True, "reason": "self_loop"}
        try:
            result = await engine.insert_edge_with_cycle_check(
                src, tgt, body.predicate,
            )
        except ValueError as e:
            return {"skipped": True, "reason": str(e)}
        return result

    # ------------------------------------------------------------------
    # KB edit endpoints
    # ------------------------------------------------------------------

    @app.post("/api/kbs", status_code=201)
    async def create_kb(body: KBCreateBody):
        try:
            await engine.create_kb(
                body.kb_id,
                name=body.name,
                description=body.description,
                topics=body.topics,
            )
        except ValueError as e:
            # Duplicate kb_id (or invalid id) — a client error, not a 500.
            raise HTTPException(status_code=409, detail=str(e))
        return {"kb_id": body.kb_id}

    @app.post("/api/kb/{kb_id}/entries", status_code=201)
    async def create_kb_entry(kb_id: str, body: KBEntryCreateBody):
        try:
            kb = await engine.open_kb(kb_id)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"KB '{kb_id}' not found")
        entry_id = await kb.write_entry(
            body.content,
            tags=body.tags,
            importance=body.importance,
        )
        return {"id": entry_id}

    @app.patch("/api/kbs/{kb_id}")
    async def patch_kb(kb_id: str, body: KBPatchBody):
        if body.archived is not None:
            await engine.set_archived(kb_id, body.archived)
        return {"ok": True}

    @app.delete("/api/kbs/{kb_id}")
    async def delete_kb(kb_id: str):
        await engine.delete_kb(kb_id)
        return {"ok": True}

    # ------------------------------------------------------------------
    # Sleep trigger
    # ------------------------------------------------------------------

    @app.post("/api/sleep")
    async def trigger_sleep(body: SleepBody):
        result = await engine.request_sleep(body.reason)
        return result

    # ------------------------------------------------------------------
    # GM-latency benchmark (engine-internal, throwaway GM)
    # ------------------------------------------------------------------

    @app.post("/api/benchmark")
    async def run_benchmark(body: BenchmarkBody):
        from krakey.engines.memory._internal.bench import (
            measure_at,
            recommend_soft_limit,
        )
        sizes = body.sizes if body.sizes is not None else [100, 500, 1000, 2000]
        results = []
        for n in sizes:
            r = await measure_at(n, dim=body.dim, query_repeats=body.repeats)
            results.append(r)
        rec = recommend_soft_limit(results)
        return {"results": results, "recommended_soft_limit": rec}

    # ------------------------------------------------------------------
    # Engine settings (read/write the engine's own settings file)
    # ------------------------------------------------------------------

    @app.get("/api/settings")
    async def get_settings():
        if not engine._config_path:
            return {}
        from krakey.engine_system.config_store import FileEngineConfigStore
        return FileEngineConfigStore("workspace").read(engine._config_path)

    @app.put("/api/settings")
    async def put_settings(payload: dict = Body(...)):
        if not engine._config_path:
            raise HTTPException(status_code=400, detail="engine has no config_path")
        from krakey.engine_system.config_store import FileEngineConfigStore
        FileEngineConfigStore("workspace").write(engine._config_path, payload)
        return {"ok": True}

    return app
