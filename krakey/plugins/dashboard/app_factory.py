"""FastAPI app factory \u2014 orchestration only.

Responsibilities:
  1. Construct the bare FastAPI instance.
  2. Attach middleware.
  3. Build concrete service adapters from `runtime` + injected args.
  4. Register each router / WS endpoint, wiring its required services.
  5. Mount /static and install the composed-template routes.

Everything mechanical lives elsewhere (middleware.py, templates.py,
services/adapters.py, routes/*.py, events/ws_route.py, web_chat/
ws_route.py); this file is the composition root \u2014 no business logic.

Test-friendly: every service arg is optional; pass your own fake to
swap out Runtime-backed behaviour in a unit test. Routes that need a
missing service are simply not registered (the endpoint is absent,
not broken).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Awaitable, Callable

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from krakey.plugins.dashboard.events.ws_route import register as _register_events_ws
from krakey.plugins.dashboard.middleware import attach_no_cache
from krakey.plugins.dashboard.routes import (
    health as _health,
    memory as _memory,
    plugins as _plugins,
    prompts as _prompts,
    settings as _settings_route,
    uploads as _uploads,
)
from krakey.plugins.dashboard.services.events import EventBroadcasterService
from krakey.plugins.dashboard.services.web_chat import WebChatService
from krakey.plugins.dashboard.services.adapters import (
    FileConfigService,
    RuntimeMemoryService,
    RuntimePluginsService,
    RuntimePromptsService,
)
from krakey.plugins.dashboard.services.memory import MemoryService
from krakey.plugins.dashboard.services.plugins import PluginsService
from krakey.plugins.dashboard.services.prompts import PromptsService
from krakey.plugins.dashboard.services.config import ConfigService
from krakey.plugins.dashboard.templates import register as _register_templates
from krakey.plugins.dashboard.web_chat.history import WebChatHistory
from krakey.plugins.dashboard.web_chat.service import RuntimeWebChatService
from krakey.plugins.dashboard.web_chat.ws_route import register as _register_chat_ws


_STATIC_DIR = Path(__file__).parent / "static"


def create_app(
    *,
    runtime: Any | None = None,
    web_chat_history: WebChatHistory | None = None,
    on_user_message: Callable[..., Awaitable[None]] | None = None,
    event_broadcaster: EventBroadcasterService | None = None,
    config_path: Path | None = None,
    on_restart: Callable[[], None] | None = None,
    plugin_configs_root: Path | str = "workspace/plugins",
    # --- overrides for unit tests (pass a fake instead of a real service) ---
    memory_service: MemoryService | None = None,
    prompts_service: PromptsService | None = None,
    plugins_service: PluginsService | None = None,
    config_service: ConfigService | None = None,
    web_chat_service: WebChatService | None = None,
) -> FastAPI:
    """Build the FastAPI app by wiring services to routers.

    `runtime` is used only to construct default service adapters; if
    you pass explicit `*_service` args, runtime can be None.
    """
    app = FastAPI(
        title="Krakey Dashboard",
        version="0.1",
        docs_url=None, redoc_url=None,
    )

    attach_no_cache(app)

    # Static assets (logo, app.js, upload pass-through).
    if _STATIC_DIR.exists():
        app.mount(
            "/static",
            StaticFiles(directory=str(_STATIC_DIR)),
            name="static",
        )

    # Composed index.html + style.css (served at `/` and `/style.css`).
    _register_templates(app)

    # --- always-on routes ---
    _health.register(app)
    _uploads.register(app)

    # --- services + routes that need them ---
    memory = memory_service or RuntimeMemoryService(runtime)
    prompts = prompts_service or RuntimePromptsService(runtime)
    plugins = plugins_service or RuntimePluginsService(
        runtime, plugin_configs_root=plugin_configs_root,
    )
    config = config_service or FileConfigService(config_path, on_restart)

    _memory.register(app, memory=memory)
    _prompts.register(app, prompts=prompts)
    _plugins.register(app, plugins=plugins)
    _settings_route.register(app, config=config)

    # --- WS endpoints (only when their backing is available) ---
    if web_chat_service is None and web_chat_history is not None:
        web_chat_service = RuntimeWebChatService(
            web_chat_history, on_user_message,
        )
    if web_chat_service is not None:
        _register_chat_ws(app, service=web_chat_service)

    if event_broadcaster is not None:
        _register_events_ws(app, broadcaster=event_broadcaster)

    return app
