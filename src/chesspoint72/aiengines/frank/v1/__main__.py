import sys

from chesspoint72.aiengines.frank.v1.engine import build_frank_controller


def main() -> int:
    controller = build_frank_controller()
    try:
        controller.start_listening_loop()
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
