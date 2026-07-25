"""Retired schema migration retained so historical commands fail safely."""


def main() -> None:
    """Prevent the legacy migration from reopening strict schemas."""

    raise SystemExit(
        "This migration is retired because it set additionalProperties=true. "
        "Use reviewed, versioned schema migrations instead."
    )


if __name__ == "__main__":
    main()
