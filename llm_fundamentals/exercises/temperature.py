"""
Exercise 2 — Sampling Parameters
================================

Goal:
    Run the same prompt at temperature 0, 0.7, and 1.2 — 5 times each — and
    document how output stability changes.

Config:
    from google import genai
    from shared.config import SETTINGS
    client = genai.Client(api_key=SETTINGS.require_api_key())

TODO: implement.
"""

from google import genai  # noqa: F401
from shared.config import SETTINGS  # noqa: F401


def main() -> None:
    raise NotImplementedError("Exercise 2 not implemented yet.")


if __name__ == "__main__":
    main()
