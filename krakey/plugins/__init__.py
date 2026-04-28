"""Built-in plugin directory.

Each subfolder is one plugin (sibling shape to ``workspace/plugins/<name>/``).
A plugin folder contains:

  * ``meta.yaml``   — manifest (name, description, components, schema)
  * Component code (``reflect.py``, ``tool.py``, ``sensory.py``,
    ``__init__.py`` factories)

User-editable plugin config lives at ``workspace/plugins/<name>/config.yaml``
regardless of whether the plugin code is built-in (here) or user-installed
(``workspace/plugins/<name>/``). Built-in plugins NEVER have ``config.yaml``
in this krakey/ directory — that would dirty the repo on user edits.

The infrastructure that *discovers* + *loads* these plugins lives in
``src.plugin_system``, NOT in this package. From the runtime's
perspective built-in and workspace plugins are identical — same
manifest format, same loader. The only difference is "ships with the
code" vs "user dropped it in".
"""
