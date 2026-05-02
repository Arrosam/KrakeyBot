"""``gui_exec`` plugin — single tool that performs GUI operations
(click / right_click / double_click / drag / type / key / screenshot)
in a target Environment.

Companion to ``cli_exec``: same architecture (Self picks env per
call, plugin must be allow-listed in the Environment Router), but
each action serializes into a ``pyautogui`` Python snippet
dispatched as ``[python, "-c", snippet]`` to ``env.run``. The same
plugin code therefore drives Windows + Linux backends uniformly —
``pyautogui`` does the platform-specific work inside the env. The
target env's Python interpreter must have ``pyautogui`` installed
(it's a project dep, so the host env has it; sandbox guests must
match).

Screenshots land at ``workspace/data/screenshots/<ts>.png`` inside
the target env's filesystem, matching the existing
``workspace/data/`` convention used by ``in_mind_note``'s state
file. The path is returned in the success Stimulus so Self can
reference it later.

The interpreter name is configurable via the plugin's
``python_cmd`` config field (default ``"python"``) so a Linux env
that only ships ``python3`` can be supported without code changes.
"""
