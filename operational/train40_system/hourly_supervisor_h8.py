"""Use the unchanged hourly recovery logic with the admitted H8 controller route."""

from operational.train40_system import hourly_supervisor


def main() -> None:
    """Accept only the existing queue's new H8 execution controller."""
    hourly_supervisor.CONTROLLER = "operational.train40_system.controller_h8_fast"
    hourly_supervisor.main()


if __name__ == "__main__":
    main()
