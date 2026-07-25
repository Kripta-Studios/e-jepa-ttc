"""Retired one-off schema rewriter retained for auditability."""


def main() -> None:
    """Refuse bulk schema mutation outside a reviewed migration."""

    raise SystemExit(
        "This one-off migration is retired. Edit and review each scientific "
        "artifact schema explicitly."
    )


if __name__ == "__main__":
    main()
