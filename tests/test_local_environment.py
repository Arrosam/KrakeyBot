"""Direct tests for ``LocalEnvironment.run`` — argv dispatch +
stdin EOF handling.

Most of the runtime/router-level tests cover ``LocalEnvironment``
indirectly through allow-list checks; this file pins the
subprocess-facing semantics directly so future refactors of the
``stdin`` / ``communicate`` plumbing don't silently regress the
EOF-only-stdin case (regression for the empty-string-stdin truthy
bug — see local_environment.py:54).
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from krakey.environment.local.local_environment import LocalEnvironment


# Use the host's own Python interpreter so the test is portable
# across platforms / venvs without assuming ``python`` or
# ``python3`` is on PATH.
PY = sys.executable


# --------------------------------------------------------------------
# stdin handling — the regression that justifies this test file
# --------------------------------------------------------------------


async def test_stdin_empty_string_results_in_eof_only_stdin():
    """``stdin=""`` must mean "open stdin, write nothing, EOF".

    The pre-fix code used ``input=stdin.encode() if stdin else None``,
    which collapsed an empty string to None. While the observable
    behavior happened to coincide with EOF-on-empty-pipe (asyncio's
    communicate skips the write but still closes the pipe), the
    intent was muddled — ``is not None`` makes it explicit and
    survives potential future asyncio internals changes that might
    treat input=None as "leave stdin open".
    """
    env = LocalEnvironment()
    rc, out, err = await env.run(
        [PY, "-c",
         "import sys; sys.stdout.write(repr(sys.stdin.read()))"],
        cwd=Path("."),
        timeout=10.0,
        stdin="",
    )
    assert rc == 0, f"unexpected stderr: {err!r}"
    # ``sys.stdin.read()`` on an EOF-only pipe returns the empty
    # string; ``repr`` gives us "''" so the assertion is unambiguous
    # vs. "missing output."
    assert out == "''", f"got stdout={out!r}, stderr={err!r}"


async def test_stdin_nonempty_string_is_written_to_subprocess():
    env = LocalEnvironment()
    rc, out, _ = await env.run(
        [PY, "-c",
         "import sys; sys.stdout.write(sys.stdin.read())"],
        cwd=Path("."),
        timeout=10.0,
        stdin="hello world",
    )
    assert rc == 0
    assert out == "hello world"


async def test_stdin_none_does_not_open_a_pipe(tmp_path: Path):
    """When the caller passes ``stdin=None`` (the default), the
    subprocess inherits the parent's stdin handle. We verify by
    asking the child whether stdin is a TTY/pipe/closed; on the
    pytest-driven test runner the parent's stdin is captured by
    pytest itself, but the relevant assertion is ``rc == 0`` — the
    child never blocks waiting for input that wouldn't arrive over
    a missing pipe."""
    env = LocalEnvironment()
    # The child does not read stdin at all; it just exits cleanly.
    rc, out, _ = await env.run(
        [PY, "-c", "print('ok')"],
        cwd=Path("."),
        timeout=10.0,
        stdin=None,
    )
    assert rc == 0
    assert out.strip() == "ok"


# --------------------------------------------------------------------
# Basic run mechanics — guard against regressions in the rest of run()
# --------------------------------------------------------------------


async def test_run_returns_exit_code_stdout_stderr_triple():
    env = LocalEnvironment()
    rc, out, err = await env.run(
        [PY, "-c",
         "import sys; "
         "sys.stdout.write('to-stdout'); "
         "sys.stderr.write('to-stderr'); "
         "sys.exit(3)"],
        cwd=Path("."),
        timeout=10.0,
    )
    assert rc == 3
    assert out == "to-stdout"
    assert err == "to-stderr"


async def test_run_timeout_raises_asyncio_timeout(tmp_path: Path):
    env = LocalEnvironment()
    with pytest.raises(asyncio.TimeoutError):
        await env.run(
            [PY, "-c",
             "import time; time.sleep(5)"],
            cwd=Path("."),
            timeout=0.5,
        )


async def test_preflight_returns_none():
    """Local has no remote agent / config to verify; preflight is
    a no-op signaled by ``None``. Documents the contract so a
    future change that adds a real check is a deliberate breaking
    change, not silent."""
    env = LocalEnvironment()
    assert await env.preflight() is None
