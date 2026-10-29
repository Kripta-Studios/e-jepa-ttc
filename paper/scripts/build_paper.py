"""Compatibility entry point for the public, portable manuscript build."""

import sys

from build_final import main

if __name__ == "__main__":
    # The final source distribution always uses the included public ledgers.
    if "--public" in sys.argv:
        sys.argv.remove("--public")
    main()
