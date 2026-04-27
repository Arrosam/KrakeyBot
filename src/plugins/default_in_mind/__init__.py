"""``default_in_mind`` plugin package.

Holds the one shared cross-component coordination key. Both the
reflect (writer) and the tentacle (reader) import ``_CACHE_KEY``
from here so neither sibling has to import the other just to learn
the contract — keeps the inter-component edge clean and avoids the
illusion of a circular dependency at the file-import layer.
"""

# Slot name in PluginContext.plugin_cache where build_reflect stashes
# its InMindReflectImpl instance for build_tentacle (loaded next) to
# pick up. See plugins/dashboard/__init__.py for the same pattern with
# WebChatHistory.
_CACHE_KEY = "in_mind_reflect"
