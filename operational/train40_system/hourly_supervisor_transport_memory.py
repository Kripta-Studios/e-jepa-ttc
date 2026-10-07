"""Use the unchanged hourly supervisor with the admitted H8 memory controller."""

from operational.train40_system import hourly_supervisor


def main() -> None:
    """Select the bounded H8 memory controller."""
    hourly_supervisor.CONTROLLER = "operational.train40_system.controller_transport_memory"
    hourly_supervisor.main()


if __name__ == "__main__":
    main()
