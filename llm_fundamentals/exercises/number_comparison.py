"""
Exercise 4 — Number Comparison Failure
======================================

Goal:
    Find a comparison or arithmetic question the model answers wrong
    confidently (e.g. "Is 9.11 greater than 9.9?") and probe whether the
    failure is tokenization or reasoning.

Config:
    from google import genai
    from shared.config import SETTINGS
    client = genai.Client(api_key=SETTINGS.require_api_key())

TODO: implement.
"""

from google import genai  # noqa: F401
from shared.config import SETTINGS  # noqa: F401


def main() -> None:
    raise NotImplementedError("Exercise 4 not implemented yet.")


if __name__ == "__main__":
    main()
