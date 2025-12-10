"""Allow running as `python -m src.engineering`."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
