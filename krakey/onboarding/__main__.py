"""``python -m krakey.onboarding`` — entry point for the wizard.

Prefer ``krakey onboard`` from the CLI; this module entry exists for
explicit ``python -m`` invocation and for the shipped subprocess
fallback.
"""
from __future__ import annotations

import sys

from krakey.onboarding.wizard import run_wizard


if __name__ == "__main__":
    try:
        run_wizard()
    except KeyboardInterrupt:
        print("\naborted.", file=sys.stderr)
        sys.exit(130)
    except EOFError:
        # Stdin closed mid-prompt (piped input ran out, terminal
        # detached). Exit cleanly rather than dumping a traceback.
        print("\naborted: stdin closed before wizard finished.",
              file=sys.stderr)
        sys.exit(1)
