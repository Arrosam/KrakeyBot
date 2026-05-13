"""Unit tests for ``ensure_instance_running``'s decision branches.

The orchestrator accepts injectable probe / runner helpers so we
exercise every branch (already-up / docker-missing / daemon-down /
container-already-running / spawn-then-wait / spawn-fails) without
touching real Docker.
"""
from __future__ import annotations

from typing import Any

from krakey.plugins.searxng_search.lifecycle import (
    ensure_instance_running,
    ensure_settings_dir,
)


def _record(target: list[Any], value: Any) -> Any:
    target.append(value)
    return value


# --------------------------------------------------------------------
# Already up — short-circuit, no docker work
# --------------------------------------------------------------------


def test_already_up_returns_true_without_docker_calls():
    docker_calls: list[str] = []

    ok = ensure_instance_running(
        instance_url="http://127.0.0.1:8888",
        docker_image="x", container_name="c", host_port=8888,
        probe=lambda h, p: True,
        docker_on_path=lambda: _record(
            docker_calls, "on_path",
        ) is not None,
        docker_daemon_alive=lambda: _record(
            docker_calls, "daemon",
        ) is not None,
        run_container=lambda *_: _record(
            docker_calls, "run",
        ) is not None,
    )

    assert ok is True
    assert docker_calls == []  # never invoked


# --------------------------------------------------------------------
# Docker missing on PATH → log + False
# --------------------------------------------------------------------


def test_docker_missing_returns_false():
    daemon_calls: list[Any] = []

    ok = ensure_instance_running(
        instance_url="http://127.0.0.1:8888",
        docker_image="x", container_name="c", host_port=8888,
        probe=lambda h, p: False,
        docker_on_path=lambda: False,
        docker_daemon_alive=lambda: _record(
            daemon_calls, "called",
        ) is not None,
    )

    assert ok is False
    # Daemon check skipped — no point if docker isn't even on PATH.
    assert daemon_calls == []


# --------------------------------------------------------------------
# Docker on PATH but daemon down (Docker Desktop not running) →
# False, NO container spawn attempted
# --------------------------------------------------------------------


def test_docker_daemon_down_returns_false_without_spawn():
    spawn_calls: list[Any] = []

    ok = ensure_instance_running(
        instance_url="http://127.0.0.1:8888",
        docker_image="x", container_name="c", host_port=8888,
        probe=lambda h, p: False,
        docker_on_path=lambda: True,
        docker_daemon_alive=lambda: False,
        container_running=lambda _: True,  # would otherwise short-circuit
        run_container=lambda *_: _record(spawn_calls, "spawn") is not None,
    )

    assert ok is False
    assert spawn_calls == []


# --------------------------------------------------------------------
# Container already running but port not yet open → just wait
# --------------------------------------------------------------------


def test_container_running_skips_spawn_and_waits():
    spawn_calls: list[Any] = []
    wait_calls: list[tuple[str, int, float]] = []

    ok = ensure_instance_running(
        instance_url="http://127.0.0.1:8888",
        docker_image="x", container_name="krakey-searxng",
        host_port=8888,
        wait_seconds=0.1,
        probe=lambda h, p: False,
        docker_on_path=lambda: True,
        docker_daemon_alive=lambda: True,
        container_running=lambda name: name == "krakey-searxng",
        run_container=lambda *_: _record(
            spawn_calls, "spawn",
        ) is not None,
        wait_for_port=lambda h, p, t: (
            wait_calls.append((h, p, t)) or True
        ),
    )

    assert ok is True
    assert spawn_calls == []
    assert wait_calls == [("127.0.0.1", 8888, 0.1)]


# --------------------------------------------------------------------
# Cold start: spawn container then wait for port
# --------------------------------------------------------------------


def test_cold_start_spawns_then_waits():
    spawn_args: list[tuple[str, str, int]] = []

    def _spawn(image: str, name: str, port: int) -> bool:
        spawn_args.append((image, name, port))
        return True

    ok = ensure_instance_running(
        instance_url="http://127.0.0.1:8888",
        docker_image="searxng/searxng:latest",
        container_name="krakey-searxng",
        host_port=8888,
        wait_seconds=0.1,
        probe=lambda h, p: False,
        docker_on_path=lambda: True,
        docker_daemon_alive=lambda: True,
        container_running=lambda _: False,
        run_container=_spawn,
        wait_for_port=lambda h, p, t: True,
    )

    assert ok is True
    assert spawn_args == [
        ("searxng/searxng:latest", "krakey-searxng", 8888),
    ]


def test_spawn_failure_returns_false():
    ok = ensure_instance_running(
        instance_url="http://127.0.0.1:8888",
        docker_image="x", container_name="c", host_port=8888,
        wait_seconds=0.1,
        probe=lambda h, p: False,
        docker_on_path=lambda: True,
        docker_daemon_alive=lambda: True,
        container_running=lambda _: False,
        run_container=lambda *_: False,
        wait_for_port=lambda h, p, t: True,  # would succeed if reached
    )

    assert ok is False


# --------------------------------------------------------------------
# Port + host parsing from instance_url
# --------------------------------------------------------------------


def test_port_parsed_from_instance_url():
    seen: list[tuple[str, int]] = []

    ensure_instance_running(
        instance_url="http://192.168.1.10:9090",
        docker_image="x", container_name="c", host_port=8888,
        wait_seconds=0.1,
        probe=lambda h, p: (seen.append((h, p)) or True),
        docker_on_path=lambda: True,
        docker_daemon_alive=lambda: True,
        container_running=lambda _: True,
        wait_for_port=lambda h, p, t: True,
    )

    assert seen == [("192.168.1.10", 9090)]


def test_ensure_settings_dir_creates_yaml_with_json_enabled(tmp_path):
    d = ensure_settings_dir(tmp_path / "searxng")
    settings = d / "settings.yml"
    assert settings.exists()
    text = settings.read_text(encoding="utf-8")
    # JSON output is the load-bearing setting — without it the
    # tool's HTTP API gets 403.
    assert "json" in text
    # Each install has a unique random secret so two boxes don't
    # share a key.
    assert "secret_key" in text
    # use_default_settings keeps SearXNG defaults for everything
    # else, so engine config etc. inherit upstream behavior.
    assert "use_default_settings: true" in text


def test_ensure_settings_dir_preserves_existing_file(tmp_path):
    d = tmp_path / "searxng"
    d.mkdir()
    custom = d / "settings.yml"
    custom.write_text(
        "use_default_settings: false\nfoo: bar\n",
        encoding="utf-8",
    )

    ensure_settings_dir(d)

    # User's custom settings remain untouched — the plugin's
    # only contract is "JSON must be enabled" and we trust the
    # operator to keep it.
    assert custom.read_text(encoding="utf-8") == (
        "use_default_settings: false\nfoo: bar\n"
    )


def test_ensure_settings_dir_creates_parent_dirs(tmp_path):
    deep = tmp_path / "a" / "b" / "c" / "searxng"
    d = ensure_settings_dir(deep)
    assert d.is_dir()
    assert (d / "settings.yml").exists()


def test_port_falls_back_to_host_port_when_url_has_none():
    """A URL without an explicit port (``http://localhost``) falls
    back to ``host_port`` for the probe + wait — useful when the
    operator points to a reverse proxy on port 80."""
    seen: list[tuple[str, int]] = []

    ensure_instance_running(
        instance_url="http://localhost",
        docker_image="x", container_name="c", host_port=8888,
        wait_seconds=0.1,
        probe=lambda h, p: (seen.append((h, p)) or True),
        docker_on_path=lambda: True,
        docker_daemon_alive=lambda: True,
        container_running=lambda _: True,
        wait_for_port=lambda h, p, t: True,
    )

    assert seen == [("localhost", 8888)]
