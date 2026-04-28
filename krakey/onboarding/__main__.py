"""``python -m src.onboarding`` — entry point for the wizard."""
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
