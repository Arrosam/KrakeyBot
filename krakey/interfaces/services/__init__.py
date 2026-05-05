"""Core service slot Protocols.

KrakeyBot has TWO axes of pluggability:

  * **Additive plugins** (``krakey.interfaces.{tool, channel, modifier}``)
    — strictly additive components users opt into via
    ``config.plugins:``. Disabling any plugin must NOT break the
    runtime's heartbeat (DevSpec invariant).

  * **Replaceable core** (this package) — built-in services the
    runtime depends on (memory, prompt builder, embedder, ...). Users
    who want to swap the default with their own implementation declare
    a dotted path in ``config.core_implementations.<slot>``; the
    runtime imports it at startup, instantiates it, and verifies it
    satisfies the slot's Protocol from this package.

Each Protocol here is paired with a slot name in
``krakey.runtime.service_resolver.ServiceResolver``. Protocols are
``@runtime_checkable`` so the resolver can do a structural
``isinstance(impl, Protocol)`` check at startup and fail loud if the
user's class is missing methods.

Adding a new slot:
  1. Add ``<slot_name>: str = ""`` to ``CoreImplementations`` in
     ``krakey/models/config/core_impls.py``
  2. Define ``<SlotName>Like(Protocol)`` in this package
  3. Update the runtime's composition root to call
     ``resolver.resolve("<slot_name>", default_factory=..., expected_protocol=...)``
"""
