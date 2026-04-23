"""Chat attachment uploads.

POST /api/chat/upload    multi-part files \u2192 stored under workspace/
GET  /uploads/<name>     serves back what was stored.

The upload dir is module-level so tests can monkey-patch it. It is
created lazily on first upload.
"""
from __future__ import annotations

from datetime import datetime as _dt
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse


# Test-monkeypatchable; see tests/test_dashboard_settings.py.
_UPLOAD_DIR = Path("workspace/data/uploads")
_MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MiB per file


def register(app: FastAPI) -> None:

    @app.post("/api/chat/upload")
    async def upload(files: list[UploadFile] = File(...)):  # noqa: ANN201
        _UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        out = []
        for f in files:
            data = await f.read()
            if len(data) > _MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"{f.filename}: exceeds {_MAX_UPLOAD_BYTES} bytes",
                )
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
