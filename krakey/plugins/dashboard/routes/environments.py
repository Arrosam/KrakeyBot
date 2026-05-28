"""Environment diagnostic status route.

- ``GET /api/environments/status`` — return a snapshot of each
  environment's (status, reason) so the dashboard can render the
  Sandbox VM section badge + the Inner Thoughts Status panel sandbox
  row on cold load (before the first EnvironmentStatusEvent arrives
  over the websocket).

The runtime's ``EnvironmentRouter.env_status()`` is the source of
truth — it survives de-registration so an env that failed preflight
can still be reported with its original failure reason.

Routes are registered unconditionally. When ``runtime`` is None the
handler returns 503 so tests can assert the contract regardless of
whether a real runtime is present.
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import JSONResponse


def register(app: FastAPI, *, runtime) -> None:

    @app.get("/api/environments/status")
    async def environments_status():  # noqa: ANN201
        if runtime is None:
            return JSONResponse(
                status_code=503,
                content={"error": "runtime not available"},
            )
        raw = runtime.environment_router.env_status()
        # ``raw`` is dict[name, (status, reason)]; reshape to the same
        # nested-dict wire shape used by ``EnvironmentStatusEvent`` so the
        # frontend's ``lastStats.env_status`` consumer treats cold-load
        # and live-event payloads identically.
        return {
            name: {"status": status, "reason": reason}
            for name, (status, reason) in raw.items()
        }
