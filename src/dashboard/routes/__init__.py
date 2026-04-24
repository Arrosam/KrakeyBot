"""REST routers \u2014 one module per API surface.

Each module exposes a `register(app, *, <services>)` function that
FastAPI decorators hang off of. `app_factory.create_app` wires
services \u2192 routers. No router takes the full Runtime: each depends
only on the narrow Protocol it needs (ISP).
"""
