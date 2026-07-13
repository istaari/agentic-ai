"""
Exercise 3 — Context Window as Working Memory
=============================================

Goal:
    Gradually fill the context window over many turns and record the
    approximate point where the model starts ignoring early instructions.

Config:
    from google import genai
    from shared.config import SETTINGS
    client = genai.Client(api_key=SETTINGS.require_api_key())

TODO: implement.
"""

from google import genai  # noqa: F401
from shared.config import SETTINGS  # noqa: F401


def main() -> None:
    raise NotImplementedError("Exercise 3 not implemented yet.")


if __name__ == "__main__":
    main()
