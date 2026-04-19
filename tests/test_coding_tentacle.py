"""Phase 3 / C: Coding tentacle (subprocess with timeout)."""
import sys
from pathlib import Path

import pytest

from src.tentacles.coding import CodingTentacle, SubprocessRunner


class FakeRunner:
    def __init__(self, batches=None, raises=None):
        self._batches = list(batches or [])
        self._raises = raises
        self.calls = []

    async def run(self, cmd, *, cwd, timeout, stdin=None):
        self.calls.append({"cmd": list(cmd), "cwd": str(cwd),
                           "timeout": timeout, "stdin": stdin})
        if self._raises is not None:
            raise self._raises
        if not self._batches:
            return 0, "", ""
        return self._batches.pop(0)


def test_metadata():
    t = CodingTentacle(runner=FakeRunner(), sandbox_dir="x")
    assert t.name == "coding"
    assert t.description
    assert t.is_internal is True


async def test_python_code_runs_via_runner(tmp_path):
    runner = FakeRunner(batches=[(0, "hello\n", "")])
    t = CodingTentacle(runner=runner, sandbox_dir=tmp_path)
    stim = await t.execute("print stuff",
                              {"language": "python",
                               "code": "print('hello')"})
    assert "hello" in stim.content
    # The runner was called with python interpreter and the code via stdin
    call = runner.calls[0]
    assert "python" in call["cmd"][0].lower()
    assert call["stdin"] == "print('hello')"
    assert call["cwd"] == str(tmp_path)


async def test_default_language_is_python(tmp_path):
    runner = FakeRunner(batches=[(0, "ok", "")])
    t = CodingTentacle(runner=runner, sandbox_dir=tmp_path)
    await t.execute("print('hi')", {})  # no language param
    assert "python" in runner.calls[0]["cmd"][0].lower()


async def test_shell_language(tmp_path):
    runner = FakeRunner(batches=[(0, "world", "")])
    t = CodingTentacle(runner=runner, sandbox_dir=tmp_path)
    stim = await t.execute("echo world",
                              {"language": "shell", "code": "echo world"})
    assert "world" in stim.content
    # On Windows we don't assume bash; runner gets the shell flag set
    # — the contract: cmd is a shell-launching command list
    assert runner.calls[0]["cmd"]


async def test_unsupported_language_returns_error(tmp_path):
    runner = FakeRunner()
    t = CodingTentacle(runner=runner, sandbox_dir=tmp_path)
    stim = await t.execute("", {"language": "brainfuck", "code": "+"})
    assert "unsupported" in stim.content.lower() or "不支持" in stim.content
    assert runner.calls == []


async def test_nonzero_return_code_in_output(tmp_path):
    runner = FakeRunner(batches=[(1, "stdout text", "stderr text")])
    t = CodingTentacle(runner=runner, sandbox_dir=tmp_path)
    stim = await t.execute("x", {"language": "python", "code": "raise"})
    assert "exit=1" in stim.content
    assert "stdout text" in stim.content
    assert "stderr text" in stim.content


async def test_timeout_returns_clear_message(tmp_path):
    import asyncio
    runner = FakeRunner(raises=asyncio.TimeoutError())
    t = CodingTentacle(runner=runner, sandbox_dir=tmp_path,
                          timeout_seconds=0.01)
    stim = await t.execute("loop", {"language": "python",
                                          "code": "while True: pass"})
    assert "timeout" in stim.content.lower() or "超时" in stim.content
    assert stim.adrenalin is True


async def test_runner_failure_returned_as_stimulus(tmp_path):
    runner = FakeRunner(raises=RuntimeError("subprocess died"))
    t = CodingTentacle(runner=runner, sandbox_dir=tmp_path)
    stim = await t.execute("x", {"language": "python", "code": "1"})
    assert "subprocess died" in stim.content


async def test_sandbox_dir_created(tmp_path):
    sb = tmp_path / "fresh_sandbox"
    assert not sb.exists()
    runner = FakeRunner(batches=[(0, "", "")])
    t = CodingTentacle(runner=runner, sandbox_dir=sb)
    await t.execute("x", {"language": "python", "code": "1"})
    assert sb.exists() and sb.is_dir()


async def test_long_output_truncated(tmp_path):
    big = "x" * 50000
    runner = FakeRunner(batches=[(0, big, "")])
    t = CodingTentacle(runner=runner, sandbox_dir=tmp_path,
                          max_output_chars=4000)
    stim = await t.execute("", {"language": "python", "code": "print"})
    assert len(stim.content) < 8000  # output capped + framing
    assert "truncated" in stim.content.lower() or "截断" in stim.content


# ---------------- SubprocessRunner real (smoke) ----------------

async def test_subprocess_runner_smoke(tmp_path):
    """Sanity check that the real SubprocessRunner can run python -c."""
    runner = SubprocessRunner()
    rc, out, err = await runner.run(
        [sys.executable, "-c", "print('smoke test ok')"],
        cwd=tmp_path, timeout=10,
    )
    assert rc == 0
    assert "smoke test ok" in out
