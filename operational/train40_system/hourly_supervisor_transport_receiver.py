"""Use frozen hourly recovery with the H8 monitor-receiver controller."""

from operational.train40_system import hourly_supervisor


def main() -> None:
    """Select only the admitted receiver controller."""

    hourly_supervisor.CONTROLLER = "operational.train40_system.controller_transport_receiver"
    hourly_supervisor.main()


if __name__ == "__main__":
    main()
