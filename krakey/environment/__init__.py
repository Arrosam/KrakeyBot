"""Environment system — transports for non-host-resident commands.

The runtime composes a single ``EnvironmentRouter`` that owns:

  * named ``Environment`` instances (currently ``LocalEnvironment``
    + ``SandboxEnvironment``; any number of impls in principle), and
  * a per-env allow-list mapping ``env_name -> [plugin_name, ...]``,
    sourced from ``config.environments``.

Plugins reach the Router via ``PluginContext.environment(env_name)``
— the per-plugin accessor checks the allow-list and returns the env
or raises ``EnvironmentDenied``. Plugins never touch the Router or
the env constructors directly.

The Protocol + exception types live one layer up in
``krakey/interfaces/environment.py`` (consistent with how
``channel`` / ``tool`` / ``modifier`` declare contracts in the
``interfaces`` package and ship impls in plugin folders).
"""
