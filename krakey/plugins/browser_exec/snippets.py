"""Pure builders + validators for the ``browser_exec`` plugin.

The plugin's runtime model (plan v2) is a long-running browser
RPC server inside the env that survives across heartbeats. Each
tool call dispatches a small Python *client snippet* via
``env.run([python_cmd, "-c", snippet])``; the snippet:

  1. Reads ``workspace/data/browser_exec/server.json`` for the
     server's port + auth token.
  2. TCP-probes the recorded port. On failure: writes the embedded
     ``server.py`` source to disk, spawns it as a detached
     subprocess, polls for ``server.json`` to reappear.
  3. POSTs ``{op, args}`` to ``http://127.0.0.1:<port>/rpc`` with
     ``X-Browser-Token: <token>``.
  4. Prints the response JSON to stdout, exits 0.

Public API:

    build_dispatch_script(op, args, *, server_source,
                          python_cmd, browser, headless,
                          rpc_timeout_s) -> str
        Return the client snippet as a Python source string. The
        op + args dict and the server source are JSON-encoded
        inside the snippet; selectors / text / URLs from Self
        travel only as JSON string values, never as bare Python
        tokens (load-bearing safety contract).

    validate_url(url) -> str
        Reject non-string / non-http(s) URLs.

    validate_action(action_dict) -> None
        Validate one in-tab action dict's shape (kind + required
        per-kind fields + types).

    SERVER_SOURCE — string holding the full source of
        ``server.py``, embedded into every dispatched snippet so
        the env doesn't need a pre-deployed copy.

    URL_SCHEMES_ALLOWED, ACTIONS, SCROLL_DIRECTIONS — module-level
        constants reused by the tool layer.
"""
from __future__ import annotations

import inspect
import json
from typing import Any

from krakey.plugins.browser_exec import server as _server_module


SERVER_SOURCE = inspect.getsource(_server_module)
"""Verbatim source of ``server.py``, embedded into every
dispatched snippet. The snippet writes this to
``<env-cwd>/workspace/data/browser_exec/server.py`` on first run
(idempotent — overwrites if the on-disk source differs from the
embedded copy, so plugin upgrades propagate without operator
intervention)."""


URL_SCHEMES_ALLOWED = ("http://", "https://")
"""Only HTTP(S) URLs cross the tool boundary. ``file://``,
``data:``, ``javascript:``, ``chrome://``, ``about:`` are rejected
— they bypass network policy or grant local-file / privileged-page
access we do not want Self to reach."""

ACTIONS = (
    "navigate", "click", "type", "press",
    "scroll", "wait_for", "screenshot",
)
"""Action kinds Self may include in the ``actions`` array."""

SCROLL_DIRECTIONS = ("up", "down", "left", "right")


def validate_url(url: Any, *, field: str = "start_url") -> str:
    """Reject non-string, non-http(s) URLs. Returns the validated
    string on success.

    The name of the field being validated is included in error
    messages so Self gets a precise pointer (``start_url`` vs
    ``actions[3].url``) when the snippet is rejected at the tool
    boundary.
    """
    if not isinstance(url, str) or not url:
        raise ValueError(
            f"`{field}` must be a non-empty string"
        )
    if not any(url.startswith(s) for s in URL_SCHEMES_ALLOWED):
        raise ValueError(
            f"`{field}` must start with http:// or https:// "
            f"(got: {url[:40]!r})"
        )
    return url


def validate_action(a: Any, *, index: int = 0) -> None:
    """Validate one action dict in-place. Raises ``ValueError`` on
    any shape problem — caller catches and converts to error
    Stimulus.

    Per-kind required fields are enforced strictly so unknown
    action kinds and missing fields are deterministic param errors,
    not runtime subprocess failures (matches the lesson learned
    from the gui_exec key-combo collapse bug)."""
    if not isinstance(a, dict):
        raise ValueError(
            f"`actions[{index}]` must be an object, got "
            f"{type(a).__name__}"
        )
    kind = a.get("action")
    if kind not in ACTIONS:
        raise ValueError(
            f"`actions[{index}].action` must be one of "
            f"{list(ACTIONS)}, got {kind!r}"
        )

    def _str_field(name: str) -> None:
        v = a.get(name)
        if not isinstance(v, str) or not v:
            raise ValueError(
                f"`actions[{index}]` action={kind!r} requires "
                f"non-empty string `{name}`"
            )

    if kind == "navigate":
        validate_url(a.get("url"), field=f"actions[{index}].url")
    elif kind == "click":
        _str_field("selector")
    elif kind == "type":
        _str_field("selector")
        # `text` may be empty string (legitimate "clear field"
        # use-case) but must be a string, not None / int / list.
        if not isinstance(a.get("text"), str):
            raise ValueError(
                f"`actions[{index}]` action='type' requires "
                f"string `text` (empty OK to clear)"
            )
    elif kind == "press":
        _str_field("key")
    elif kind == "scroll":
        if a.get("direction") not in SCROLL_DIRECTIONS:
            raise ValueError(
                f"`actions[{index}]` action='scroll' requires "
                f"`direction` in {list(SCROLL_DIRECTIONS)}, got "
                f"{a.get('direction')!r}"
            )
        amt = a.get("amount")
        if (
            not isinstance(amt, (int, float))
            or isinstance(amt, bool)
            or amt <= 0
        ):
            raise ValueError(
                f"`actions[{index}]` action='scroll' requires "
                f"positive number `amount`"
            )
    elif kind == "wait_for":
        _str_field("selector")
        if "timeout_ms" in a:
            tm = a["timeout_ms"]
            if (
                not isinstance(tm, (int, float))
                or isinstance(tm, bool)
                or tm <= 0
            ):
                raise ValueError(
                    f"`actions[{index}]` action='wait_for' "
                    f"`timeout_ms` must be a positive number"
                )
    elif kind == "screenshot":
        if "full_page" in a and not isinstance(a["full_page"], bool):
            raise ValueError(
                f"`actions[{index}]` action='screenshot' "
                f"`full_page` must be a boolean if provided"
            )


# The dispatch-client snippet template. All caller-controlled
# values (``op``, ``args``, server config) travel through three
# JSON literals — no f-string interpolation, no bare-token leaks.
# Substitutions: {payload_literal}, {server_source_literal},
# {browser_literal}, {headless_literal}, {rpc_timeout_s},
# {python_cmd_literal}.
#
# Doubled braces (``{{ ... }}``) survive ``.format()`` as single
# braces in the emitted source.
_DISPATCH_TEMPLATE = """\
import json, os, socket, subprocess, sys, time
from pathlib import Path

PAYLOAD = json.loads({payload_literal})
SERVER_SOURCE = json.loads({server_source_literal})
BROWSER = {browser_literal}
HEADLESS = {headless_literal}
RPC_TIMEOUT_S = {rpc_timeout_s}
PYTHON_CMD = {python_cmd_literal}

WORKSPACE = Path('workspace')
DATA_DIR = WORKSPACE / 'data' / 'browser_exec'
INFO_PATH = DATA_DIR / 'server.json'
SERVER_PATH = DATA_DIR / 'server.py'


def _read_info():
    try:
        return json.loads(INFO_PATH.read_text(encoding='utf-8'))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _server_alive(info):
    if not info:
        return False
    try:
        with socket.create_connection(
            ('127.0.0.1', int(info['port'])), timeout=0.5,
        ):
            return True
    except (OSError, ValueError):
        return False


def _write_server_source():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if SERVER_PATH.exists():
        try:
            existing = SERVER_PATH.read_text(encoding='utf-8')
            if existing == SERVER_SOURCE:
                return
        except OSError:
            pass
    SERVER_PATH.write_text(SERVER_SOURCE, encoding='utf-8')


def _spawn_server():
    _write_server_source()
    args = [
        PYTHON_CMD, str(SERVER_PATH),
        '--workspace', str(WORKSPACE),
        '--browser', BROWSER,
        '--headless', 'true' if HEADLESS else 'false',
    ]
    kwargs = {{
        'stdin': subprocess.DEVNULL,
        'stdout': subprocess.DEVNULL,
        'stderr': subprocess.DEVNULL,
        'cwd': str(Path('.').resolve()),
        'close_fds': True,
    }}
    if os.name == 'nt':
        # Windows: detach from console + new process group so
        # the parent's exit doesn't take the child down.
        DETACHED = 0x00000008  # DETACHED_PROCESS
        NEW_PG = 0x00000200    # CREATE_NEW_PROCESS_GROUP
        kwargs['creationflags'] = DETACHED | NEW_PG
    else:
        # POSIX: new session reparents to PID 1 when we exit.
        kwargs['start_new_session'] = True
    subprocess.Popen(args, **kwargs)


def _wait_for_ready(deadline):
    while time.time() < deadline:
        info = _read_info()
        if _server_alive(info):
            return info
        time.sleep(0.2)
    return None


def _post_rpc(info, body):
    import http.client
    conn = http.client.HTTPConnection(
        '127.0.0.1', int(info['port']), timeout=RPC_TIMEOUT_S,
    )
    raw = json.dumps(body).encode('utf-8')
    conn.request(
        'POST', '/rpc', body=raw,
        headers={{
            'Content-Type': 'application/json',
            'Content-Length': str(len(raw)),
            'X-Browser-Token': info['token'],
        }},
    )
    resp = conn.getresponse()
    data = resp.read().decode('utf-8', errors='replace')
    conn.close()
    return resp.status, data


def main():
    info = _read_info()
    if not _server_alive(info):
        try:
            INFO_PATH.unlink(missing_ok=True)
        except Exception:
            pass
        _spawn_server()
        info = _wait_for_ready(time.time() + 30.0)
        if info is None:
            log_path = DATA_DIR / 'server.log'
            tail = ''
            try:
                tail = '\\n'.join(
                    log_path.read_text(encoding='utf-8').splitlines()[-40:]
                )
            except OSError:
                tail = '(no server.log yet)'
            sys.stdout.write(json.dumps({{
                'ok':    False,
                'error': 'browser server failed to start within 30s',
                'tabs':  [],
                'log_tail': tail,
            }}))
            return 0

    status, body = _post_rpc(info, PAYLOAD)
    if status != 200:
        sys.stdout.write(json.dumps({{
            'ok':    False,
            'error': 'rpc http status ' + str(status) + ': ' + body[:400],
            'tabs':  [],
        }}))
        return 0

    sys.stdout.write(body)
    return 0


sys.exit(main())
"""


def build_dispatch_script(
    op: str,
    args: dict[str, Any],
    *,
    python_cmd: str,
    browser: str,
    headless: bool,
    rpc_timeout_s: float = 60.0,
) -> str:
    """Build the Python source for one RPC dispatch.

    The op name + args dict travel as JSON inside the snippet; the
    server source travels as a JSON-encoded Python string literal.
    The snippet's only Python-source-level vars are
    BROWSER / HEADLESS / RPC_TIMEOUT_S / PYTHON_CMD which are tool-
    layer constants, never Self-controlled. All Self-controlled
    values (selectors, URLs, text) are inside the PAYLOAD dict.
    """
    payload = {"op": op, "args": args}

    def _python_str_literal(s: str) -> str:
        # Outer json.dumps wraps a JSON string in a Python-safe
        # string literal (escapes quotes, backslashes, control
        # chars). The emitted snippet does json.loads(<literal>)
        # to recover the value.
        return json.dumps(json.dumps(s, ensure_ascii=False))

    payload_json = json.dumps(payload, ensure_ascii=False)

    return _DISPATCH_TEMPLATE.format(
        payload_literal=json.dumps(payload_json),
        server_source_literal=_python_str_literal(SERVER_SOURCE),
        browser_literal=repr(browser),
        headless_literal=repr(bool(headless)),
        rpc_timeout_s=float(rpc_timeout_s),
        python_cmd_literal=repr(python_cmd),
    )
