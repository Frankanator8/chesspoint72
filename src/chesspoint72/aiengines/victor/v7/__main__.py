import sys
from chesspoint72.aiengines.victor.v7.engine import build_controller

def main() -> int:
    controller = build_controller()
    try:
        controller.start_listening_loop()
    except KeyboardInterrupt:
        return 130
    return 0

if __name__ == "__main__":
    sys.exit(main())
