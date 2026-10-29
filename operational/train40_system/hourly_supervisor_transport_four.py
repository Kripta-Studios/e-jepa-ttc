"""Use the unchanged hourly supervisor with the admitted H8 four-worker controller."""

from operational.train40_system import hourly_supervisor


def main() -> None:
    """Select the bounded H8 four-worker controller."""
    hourly_supervisor.CONTROLLER = "operational.train40_system.controller_transport_four"
    hourly_supervisor.main()


if __name__ == "__main__":
    main()
