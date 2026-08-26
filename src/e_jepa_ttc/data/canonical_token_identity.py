"""Single canonical ordered-token identity for Scientific Recovery V8.

Frozen V8 evidence uses two named hash fields. They are not interchangeable:

- ``train_sample_tokens_sha256`` / ``dev_sample_tokens_sha256`` /
  ``sorted_sample_tokens_sha256`` are length-prefixed sorted token strings.
- ``ordered_token_ids_sha256`` (and sibling row/target/weight/fold hashes) are
  newline-delimited canonical JSON records sorted by ``token_id``.

Every producer and consumer of those fields must call the matching function.
A mismatch is a hard integrity failure, never a warning.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence

from e_jepa_ttc.artifacts.hashing import canonical_json

SCHEMA_VERSION = "scientific_recovery_v8_token_identity_v1"
SORTED_TOKEN_STRING_ALGORITHM = (
    "sha256(utf-8 token strings sorted lexicographically; each encoded as "
    "uint64-be length prefix followed by bytes)"
)
CANONICAL_JSON_RECORDS_ALGORITHM = (
    "sha256(canonical JSON records sorted by token_id, newline-delimited UTF-8)"
)


def hash_sorted_token_strings(tokens: Iterable[str]) -> str:
    """Hash a fold or universe of sample-token strings.

    This is the frozen V5/V8 algorithm for ``train_sample_tokens_sha256``,
    ``dev_sample_tokens_sha256``, and ``sorted_sample_tokens_sha256``.
    """

    digest = hashlib.sha256()
    for value in sorted(str(item) for item in tokens):
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def hash_canonical_json_records(
    records: Sequence[Mapping[str, str]],
    *,
    sort_key: str = "token_id",
) -> str:
    """Hash newline-delimited canonical JSON records in lexicographic token order.

    This is the frozen V8 algorithm for ``ordered_token_ids_sha256`` and the
    other sample-contract hashes emitted by the protocol freezer.
    """

    ordered = sorted((dict(record) for record in records), key=lambda record: str(record[sort_key]))
    payload = b"".join(canonical_json(record) + b"\n" for record in ordered)
    return hashlib.sha256(payload).hexdigest()


def hash_ordered_token_ids(token_ids: Iterable[str]) -> str:
    """Hash the frozen ``ordered_token_ids_sha256`` field from token strings."""

    return hash_canonical_json_records([{"token_id": str(token)} for token in token_ids])
