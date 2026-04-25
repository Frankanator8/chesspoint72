import sys

from chesspoint72.aiengines.frank.v2 import build_frank_v2


def main() -> int:
    controller = build_frank_v2()
    try:
        controller.start_listening_loop()
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())

