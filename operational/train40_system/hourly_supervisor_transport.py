"""Use the frozen hourly recovery logic with the H8 transport controller."""

from operational.train40_system import hourly_supervisor


def main() -> None:
    """Accept only the admitted H8 transport controller route."""

    hourly_supervisor.CONTROLLER = "operational.train40_system.controller_transport_graph"
    hourly_supervisor.main()


if __name__ == "__main__":
    main()
