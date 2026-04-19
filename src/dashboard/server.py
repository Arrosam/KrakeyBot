"""Phase 3.F.2: FastAPI dashboard server.

Skeleton: app factory + DashboardServer that starts/stops uvicorn
programmatically inside the runtime's event loop. Subsequent sub-phases
add WS feeds, REST endpoints, and the SPA assets.
"""
from __future__ import annotations

import asyncio
import socket
from pathlib import Path
from typing import Any, Awaitable, Callable

import uvicorn
from fastapi import (FastAPI, File, HTTPException, Query, UploadFile,
                       WebSocket, WebSocketDisconnect)
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from src.dashboard.events_ws import EventBroadcaster
from src.dashboard.web_chat import WebChatHistory


_STATIC_DIR = Path(__file__).parent / "static"


def create_app(
    *,
    runtime: Any | None,
    web_chat_history: WebChatHistory | None = None,
    on_user_message: Callable[[str], Awaitable[None]] | None = None,
    event_broadcaster: EventBroadcaster | None = None,
    config_path: Path | None = None,
    on_restart: Callable[[], None] | None = None,
) -> FastAPI:
    """Build the FastAPI app.

    `runtime` is the live Runtime (used by REST endpoints, may be None in
    skeleton tests). `web_chat_history` + `on_user_message` are needed for
    the /ws/chat endpoint (also optional for narrow tests).
    """
    app = FastAPI(title="Krakey Dashboard",
                    version="0.1",
                    docs_url=None, redoc_url=None)

    if _STATIC_DIR.exists():
        app.mount("/static",
                    StaticFiles(directory=str(_STATIC_DIR)),
                    name="static")

    @app.get("/api/health")
    async def health():  # noqa: ANN201
        return {"status": "ok"}

    @app.get("/", response_class=HTMLResponse)
    async def index():  # noqa: ANN201
        index_path = _STATIC_DIR / "index.html"
        if not index_path.exists():
            return HTMLResponse(_FALLBACK_INDEX, status_code=200)
        return FileResponse(index_path, media_type="text/html")

    if web_chat_history is not None:
        _attach_chat_ws(app, web_chat_history, on_user_message)

    if event_broadcaster is not None:
        _attach_events_ws(app, event_broadcaster)

    _attach_memory_routes(app, runtime)
    _attach_settings_routes(app, config_path, on_restart)
    _attach_upload_route(app)

    return app


_UPLOAD_DIR = Path("workspace/data/uploads")
_MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MiB per file


def _attach_upload_route(app: FastAPI) -> None:
    @app.post("/api/chat/upload")
    async def upload(files: list[UploadFile] = File(...)):  # noqa: ANN201
        _UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        from datetime import datetime as _dt
        out = []
        for f in files:
            data = await f.read()
            if len(data) > _MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413,
                    detail=f"{f.filename}: exceeds {_MAX_UPLOAD_BYTES} bytes")
            stamp = _dt.now().strftime("%Y%m%d-%H%M%S-%f")
            safe = "".join(c for c in (f.filename or "file")
                           if c.isalnum() or c in "._-")
            dest = _UPLOAD_DIR / f"{stamp}_{safe}"
            dest.write_bytes(data)
            out.append({
                "name": f.filename or safe,
                "url": f"/uploads/{dest.name}",
                "type": f.content_type or "application/octet-stream",
                "size": len(data),
            })
        return {"files": out}

    @app.get("/uploads/{filename}")
    async def serve_upload(filename: str):  # noqa: ANN201
        path = _UPLOAD_DIR / filename
        if not path.exists() or not path.is_file():
            raise HTTPException(status_code=404, detail="not found")
        return FileResponse(path)


# ---------------- Memory browser (read-only) ----------------


def _attach_memory_routes(app: FastAPI, runtime: Any | None) -> None:
    def _runtime_or_503():
        if runtime is None or not hasattr(runtime, "gm"):
            raise HTTPException(status_code=503,
                                  detail="runtime not available")
        return runtime

    @app.get("/api/gm/nodes")
    async def gm_nodes(
        category: str | None = None,
        limit: int = Query(default=200, ge=1, le=2000),
    ):  # noqa: ANN201
        rt = _runtime_or_503()
        nodes = await rt.gm.list_nodes(category=category, limit=limit)
        return {"count": len(nodes),
                "nodes": [_serialize_node(n) for n in nodes]}

    @app.get("/api/gm/edges")
    async def gm_edges(limit: int = Query(default=500, ge=1, le=5000)):  # noqa: ANN201
        rt = _runtime_or_503()
        # Pull all edges (Phase-1 small scale); cap with limit on output
        db = rt.gm._require()  # noqa: SLF001
        async with db.execute(
            "SELECT na.name AS source, e.predicate AS predicate, "
            "nb.name AS target FROM gm_edges e "
            "JOIN gm_nodes na ON na.id=e.node_a "
            "JOIN gm_nodes nb ON nb.id=e.node_b "
            "ORDER BY e.id ASC LIMIT ?", (limit,),
        ) as cur:
            rows = await cur.fetchall()
        edges = [{"source": r["source"], "target": r["target"],
                  "predicate": r["predicate"]} for r in rows]
        return {"count": len(edges), "edges": edges}

    @app.get("/api/gm/stats")
    async def gm_stats():  # noqa: ANN201
        rt = _runtime_or_503()
        total_nodes = await rt.gm.count_nodes()
        total_edges = await rt.gm.count_edges()
        db = rt.gm._require()  # noqa: SLF001
        async with db.execute(
            "SELECT category, COUNT(*) FROM gm_nodes GROUP BY category"
        ) as cur:
            cat_rows = await cur.fetchall()
        async with db.execute(
            "SELECT source_type, COUNT(*) FROM gm_nodes GROUP BY source_type"
        ) as cur:
            src_rows = await cur.fetchall()
        return {
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "by_category": {r[0]: r[1] for r in cat_rows},
            "by_source": {r[0]: r[1] for r in src_rows},
        }

    @app.get("/api/kbs")
    async def kbs():  # noqa: ANN201
        rt = _runtime_or_503()
        return {"kbs": await rt.kb_registry.list_kbs()}

    @app.get("/api/kb/{kb_id}/entries")
    async def kb_entries(kb_id: str,
                            limit: int = Query(default=200, ge=1, le=2000)):  # noqa: ANN201
        rt = _runtime_or_503()
        try:
            kb = await rt.kb_registry.open_kb(kb_id)
        except KeyError:
            raise HTTPException(status_code=404,
                                  detail=f"KB {kb_id!r} not found")
        db = kb._require()  # noqa: SLF001
        async with db.execute(
            "SELECT id, content, source, tags, importance, created_at "
            "FROM kb_entries WHERE is_active = 1 ORDER BY id DESC LIMIT ?",
            (limit,),
        ) as cur:
            rows = await cur.fetchall()
        import json as _json
        entries = []
        for r in rows:
            tags = _json.loads(r["tags"]) if r["tags"] else []
            entries.append({"id": r["id"], "content": r["content"],
                              "source": r["source"], "tags": tags,
                              "importance": r["importance"],
                              "created_at": r["created_at"]})
        return {"kb_id": kb_id, "count": len(entries), "entries": entries}


def _serialize_node(n: dict[str, Any]) -> dict[str, Any]:
    """Trim raw embedding from API response; keep its presence as a flag."""
    out = {k: v for k, v in n.items() if k != "embedding"}
    out["has_embedding"] = n.get("embedding") is not None
    return out


# ---------------- Settings (read + write + restart) ----------------


def _attach_settings_routes(app: FastAPI, config_path: Path | None,
                              on_restart: Callable[[], None] | None) -> None:
    from fastapi import Body
    import yaml as _yaml

    from src.models.config_backup import backup_config

    @app.get("/api/settings")
    async def get_settings():  # noqa: ANN201
        if config_path is None:
            raise HTTPException(status_code=503,
                                  detail="config_path not provided")
        if not Path(config_path).exists():
            raise HTTPException(status_code=404,
                                  detail=f"config not found: {config_path}")
        raw = Path(config_path).read_text(encoding="utf-8")
        try:
            parsed = _yaml.safe_load(raw)
        except _yaml.YAMLError:
            parsed = None
        return {"path": str(config_path), "raw": raw, "parsed": parsed}

    @app.post("/api/settings")
    async def post_settings(payload: dict = Body(...)):  # noqa: ANN201
        if config_path is None:
            raise HTTPException(status_code=503,
                                  detail="config_path not provided")
        # Accept either structured `parsed` (new form-based) or `raw` YAML.
        if "parsed" in payload and payload["parsed"] is not None:
            try:
                new_raw = _yaml.safe_dump(payload["parsed"], allow_unicode=True,
                                           sort_keys=False)
            except _yaml.YAMLError as e:
                raise HTTPException(status_code=400,
                                      detail=f"cannot serialize: {e}")
        else:
            new_raw = payload.get("raw")
            if new_raw is None or not isinstance(new_raw, str):
                raise HTTPException(status_code=400,
                                      detail="missing 'parsed' or 'raw' field")
            try:
                _yaml.safe_load(new_raw)
            except _yaml.YAMLError as e:
                raise HTTPException(status_code=400,
                                      detail=f"invalid YAML: {e}")
        backup_dir = payload.get("backup_dir") or "workspace/backups"
        backup_path = backup_config(config_path, backup_dir)
        Path(config_path).write_text(new_raw, encoding="utf-8")
        return {
            "status": "saved",
            "backup": str(backup_path) if backup_path else None,
            "restart_required": True,
        }

    @app.post("/api/restart")
    async def restart():  # noqa: ANN201
        if on_restart is None:
            raise HTTPException(status_code=503,
                                  detail="restart not wired")
        try:
            on_restart()
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500,
                                  detail=f"restart failed: {e}")
        return {"status": "restarting"}


def _attach_events_ws(app: FastAPI, broadcaster: EventBroadcaster) -> None:

    @app.websocket("/ws/events")
    async def events_ws(ws: WebSocket):  # noqa: ANN201
        await ws.accept()
        await ws.send_json({"kind": "history",
                              "events": broadcaster.recent()})

        async def _send(msg):
            await ws.send_json(msg)

        broadcaster.add_socket(_send)
        try:
            while True:
                # Just keep the socket alive; events are server-pushed.
                await ws.receive_text()
        except WebSocketDisconnect:
            pass
        except Exception:  # noqa: BLE001
            pass
        finally:
            broadcaster.remove_socket(_send)


def _attach_chat_ws(
    app: FastAPI,
    history: WebChatHistory,
    on_user_message: Callable[[str], Awaitable[None]] | None,
) -> None:

    @app.websocket("/ws/chat")
    async def chat_ws(ws: WebSocket):  # noqa: ANN201
        await ws.accept()
        # 1. push full chat history on connect
        await ws.send_json({"kind": "history",
                              "messages": history.all_messages()})

        # 2. subscribe to live broadcasts
        async def _send(msg):
            try:
                await ws.send_json({"kind": "message", "message": msg})
            except Exception:  # noqa: BLE001
                # client gone — let the recv loop catch it on next round
                pass

        history.subscribe(_send)
        try:
            while True:
                data = await ws.receive_json()
                text = (data.get("text") or "").strip()
                attachments = data.get("attachments") or []
                if not text and not attachments:
                    continue
                await history.append("user", text, attachments=attachments)
                if on_user_message is not None:
                    try:
                        await on_user_message(text, attachments)
                    except Exception:  # noqa: BLE001
                        pass
        except WebSocketDisconnect:
            pass
        except Exception:  # noqa: BLE001
            pass
        finally:
            history.unsubscribe(_send)


class DashboardServer:
    """Run uvicorn in the same asyncio loop as the runtime.

    Use port=0 to let the OS pick an ephemeral port (handy for tests);
    after `.start()`, `self.port` reflects the actually-bound port.
    """

    def __init__(self, app: FastAPI, *, host: str = "127.0.0.1",
                  port: int = 8765, log_level: str = "warning"):
        self._app = app
        self._host = host
        self._requested_port = port
        self._log_level = log_level
        self._server: uvicorn.Server | None = None
        self._task: asyncio.Task | None = None
        self.port: int | None = None

    async def start(self) -> None:
        # Pre-bind to detect port-in-use loud-and-early. uvicorn's own
        # bind happens deep in its serve loop; we want the OSError now.
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
        try:
            sock.bind((self._host, self._requested_port))
        except OSError:
            sock.close()
            raise
        self.port = sock.getsockname()[1]
        # Hand the bound socket to uvicorn so we keep the port without
        # a TOCTOU window.
        config = uvicorn.Config(
            self._app, log_level=self._log_level,
            access_log=False, lifespan="off",
        )
        self._server = uvicorn.Server(config)
        # uvicorn's serve(sockets=...) takes pre-bound sockets
        self._task = asyncio.create_task(self._server.serve(sockets=[sock]))
        # Wait for it to come up
        for _ in range(50):
            await asyncio.sleep(0.02)
            if self._server.started:
                return
        # If we get here, server didn't start in time but task may still be alive
        if self._task.done() and self._task.exception() is not None:
            raise self._task.exception()  # type: ignore[misc]

    async def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except asyncio.TimeoutError:
                self._task.cancel()
                try:
                    await self._task
                except (asyncio.CancelledError, Exception):
                    pass
            self._task = None
            self._server = None


_FALLBACK_INDEX = """<!doctype html>
<html lang="zh">
<head>
  <meta charset="utf-8">
  <title>Krakey Dashboard</title>
</head>
<body>
  <h1>Krakey Dashboard</h1>
  <p>Static assets not built yet. The API is up at <code>/api/health</code>.</p>
</body>
</html>
"""
