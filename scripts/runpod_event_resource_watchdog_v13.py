"""V12 integrity/recovery checks with an explicitly authorized deadline extension."""

import hashlib
import importlib.util
import os
import sys
from pathlib import Path


def main() -> None:
    source = Path(__file__).resolve()
    old = source.with_name("runpod_event_resource_watchdog_v12.py")
    assert (
        hashlib.sha256(old.read_bytes()).hexdigest()
        == "06a453b3facd3db2dc55a0356b8705e91daaf2da49a7ce8c737932bf04bd6944"
    )
    overlay = source.parent / "resume_20261027/runpod_deadline_extension_20261027.py"
    assert (
        hashlib.sha256(overlay.read_bytes()).hexdigest()
        == os.environ["EJEPA_DEADLINE_OVERLAY_SHA256"]
    )
    sys.path.insert(0, str(overlay.parent))
    from runpod_deadline_extension_20261027 import install_watchdog

    spec = importlib.util.spec_from_file_location("ejepa_watchdog_v12_extended", old)
    wrapper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wrapper)
    module = wrapper.install_v11()
    install_watchdog(module)
    module.__file__ = str(source)
    return module.main()


if __name__ == "__main__":
    raise SystemExit(main())
