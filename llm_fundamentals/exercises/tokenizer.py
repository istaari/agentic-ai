"""
Exercise 1 — Tokenization
=========================

Goal:
    Use a tokenizer to count tokens for ~10 inputs and find cases where short
    text costs more tokens than expected (emoji, code, non-English, numbers).

Config:
    from google import genai
    from shared.config import SETTINGS
    client = genai.Client(api_key=SETTINGS.require_api_key())

TODO: implement.
"""

from google import genai  # noqa: F401
from shared.config import SETTINGS  # noqa: F401


def main() -> None:
    raise NotImplementedError("Exercise 1 not implemented yet.")


if __name__ == "__main__":
    main()
