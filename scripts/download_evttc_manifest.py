"""Plan EvTTC starter retrieval without downloading data by default.

The implementation remains in ``download_evttc_starter.py``; this stable name
exists for the AGENTS.md CLI layout and preserves its explicit ``--execute``
opt-in.
"""

from __future__ import annotations

from download_evttc_starter import main

if __name__ == "__main__":
    raise SystemExit(main())
