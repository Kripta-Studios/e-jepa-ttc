"""Correct wire newlines without modifying the manifest-bound v1 control bundle."""

from __future__ import annotations

if __package__:
    from . import runpod_event_transfer_v1 as transfer_v1
else:
    import runpod_event_transfer_v1 as transfer_v1

# v1 embeds Python source in a raw string. Its double escape emits a literal
# backslash-n on the wire, leaving the sender blocked in readline().
transfer_v1.REMOTE_RECEIVER = transfer_v1.REMOTE_RECEIVER.replace(r'"\\n"', r'"\n"')


if __name__ == "__main__":
    raise SystemExit(transfer_v1.main())
