import sys

from src.gui import app


def main() -> int:
    instance = app.Application()

    return instance.run()


if __name__ == "__main__":
    sys.exit(main())
