"""``cli_exec`` plugin — single tool that dispatches CLI argv through
the Environment Router.

Self picks the target environment per call via the ``env`` parameter;
the plugin must be allow-listed in
``config.environments.<env>.allowed_plugins`` for each env it should
be able to use. The tool catches ``EnvironmentDenied`` /
``EnvironmentUnavailableError`` / ``asyncio.TimeoutError`` and returns
an error ``Stimulus`` instead of letting them propagate, preserving
the additive-plugin invariant.
"""
