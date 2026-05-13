"""Unit tests for ``check_install.main`` — the post_install Docker
probe. Verifies the warning fires when ``docker`` is missing from
PATH and the success path stays silent on stderr.
"""
from __future__ import annotations

import pytest

from krakey.plugins.searxng_search import check_install


def test_returns_zero_when_docker_on_path(monkeypatch, capsys):
    monkeypatch.setattr(
        check_install.shutil, "which",
        lambda name: "/usr/bin/docker" if name == "docker" else None,
    )

    rc = check_install.main()

    assert rc == 0
    out = capsys.readouterr()
    # No stderr noise on the happy path — keeps `krakey install`
    # output clean for operators who already have Docker.
    assert out.err == ""
    assert "docker found" in out.out


def test_returns_one_when_docker_missing(monkeypatch, capsys):
    monkeypatch.setattr(
        check_install.shutil, "which", lambda _name: None,
    )

    rc = check_install.main()

    assert rc == 1
    err = capsys.readouterr().err
    # Message names BOTH escape paths so the operator knows the
    # plugin still works without Docker (just without auto_start).
    assert "docker" in err.lower()
    assert "auto_start" in err.lower()
    assert "instance_url" in err.lower()


@pytest.mark.parametrize("which_returns", ["", None])
def test_treats_empty_or_none_as_missing(monkeypatch, which_returns):
    monkeypatch.setattr(
        check_install.shutil, "which", lambda _name: which_returns,
    )
    # ``shutil.which`` officially returns path-or-None, but a corrupted
    # PATH could yield ""; both should surface the warning so the
    # operator isn't silently told "docker found" when it isn't.
    assert check_install.main() == 1
