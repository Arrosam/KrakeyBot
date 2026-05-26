"""``EnvironmentRouter`` — owns named ``Environment`` instances + the
plugin → env allow-list, and resolves per-plugin requests.

Two responsibilities, deliberately fused:

  * **Registry** — ``env_name -> Environment`` instance, populated
    once at runtime startup from ``config.environments``.
  * **Authorization** — per-env allow-list. ``for_plugin(plugin,
    env)`` returns the env iff the plugin is allow-listed; otherwise
    raises ``EnvironmentDenied``.

The Router itself is **construction-and-lookup-only**. It does not
import any concrete env class — Runtime builds the env instances
and hands them in. Keeps Router insulated from impl details so a
hypothetical third env (browser? web?) plugs in without touching
this file.

Lazy-call-time semantics. A plugin that never asks for an env never
trips the Router. This preserves the zero-plugin-runtime invariant
(see ``CLAUDE.md`` + ``tests/test_zero_plugin_runtime.py``): an
empty Router is a no-op.
"""
from __future__ import annotations

import logging
from typing import Any

from krakey.interfaces.environment import (
    Environment, EnvironmentDenied, EnvironmentUnavailableError,
)

_log = logging.getLogger(__name__)


class EnvironmentRouter:
    def __init__(
        self,
        envs: dict[str, Environment] | None = None,
        allow_list: dict[str, list[str]] | None = None,
    ):
        self._envs: dict[str, Environment] = dict(envs or {})
        # Stored as set per env for O(1) membership checks; the
        # config layer hands lists in.
        self._allow: dict[str, set[str]] = {
            env_name: set(plugins or [])
            for env_name, plugins in (allow_list or {}).items()
        }

    # ---- read surface ------------------------------------------------

    def env_names(self) -> list[str]:
        """All registered env names (insertion order)."""
        return list(self._envs.keys())

    def is_empty(self) -> bool:
        """True iff no envs registered. Empty Router = no-op."""
        return not self._envs

    # ---- per-plugin dispatch -----------------------------------------

    def for_plugin(self, plugin_name: str, env_name: str) -> Environment:
        """Resolve ``env_name`` for ``plugin_name``.

        Raises ``EnvironmentDenied`` when:
          * ``env_name`` is not registered (config doesn't define it), OR
          * ``plugin_name`` is not in that env's allow-list.

        Both cases are denial from the plugin's POV — the plugin
        cannot use that env, regardless of why. Distinct error
        messages so misconfiguration is debuggable.
        """
        if env_name not in self._envs:
            raise EnvironmentDenied(
                f"plugin {plugin_name!r} requested environment "
                f"{env_name!r}, but no such environment is configured. "
                f"Configured: {sorted(self._envs.keys()) or '(none)'}."
            )
        allowed = self._allow.get(env_name, set())
        if plugin_name not in allowed:
            raise EnvironmentDenied(
                f"plugin {plugin_name!r} is not allow-listed for "
                f"environment {env_name!r}. Add it to "
                f"`environments.{env_name}.allowed_plugins` in "
                f"config.yaml."
            )
        return self._envs[env_name]

    # ---- preflight ---------------------------------------------------

    async def preflight_all(self) -> list[dict[str, Any]]:
        """Walk every env that has at least one allow-listed plugin
        and call its ``preflight()``. Returns the list of non-None
        info payloads (one per env that returned readiness data).

        One env's preflight failure does NOT abort the others and
        does NOT abort startup. An env whose ``preflight()`` raises
        ``EnvironmentUnavailableError`` is de-registered: removed
        from ``_envs`` and ``_allow`` after the loop completes. A
        plugin that later targets the dropped env via ``for_plugin``
        receives ``EnvironmentDenied`` ("no such environment") —
        treated as not-configured. This lets the runtime start
        normally when only a sandbox is unreachable (local env still
        works). Both the failure and the de-registration are logged
        at warning level.
        """
        infos: list[dict[str, Any]] = []
        failed_names: list[str] = []
        for env_name, env in self._envs.items():
            if not self._allow.get(env_name):
                continue  # no plugins use this env; skip preflight
            try:
                info = await env.preflight()
            except EnvironmentUnavailableError as e:
                _log.warning(
                    "environment %r preflight failed: %s", env_name, e,
                )
                _log.warning(
                    "environment %r de-registered; plugins targeting it "
                    "will receive EnvironmentDenied",
                    env_name,
                )
                failed_names.append(env_name)
                continue
            if info is not None:
                infos.append({"env": env_name, **info})
        # De-register after the loop — never mutate _envs while iterating.
        for name in failed_names:
            self._envs.pop(name)
            self._allow.pop(name, None)
        return infos
