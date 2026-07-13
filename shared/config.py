from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

# Load the project-root .env regardless of the current working directory,
# so exercises run correctly from any section folder. Root is the parent
# of the `shared/` package this file lives in.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Settings:
    """Static configuration shared across every exercise."""

    # The Gemini model to use. Swap freely while experimenting.
    #   - gemini-2.5-flash : fast + cheap, good default for exercises
    #   - gemini-2.5-pro   : stronger reasoning, slower + costlier
    model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    # API key is read from the environment; kept out of source control.
    api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))

    # Default sampling parameters. Individual exercises override these
    # (e.g. exercise 2 sweeps temperature explicitly).
    temperature: float = float(os.getenv("GEMINI_TEMPERATURE", "0.7"))
    top_p: float = float(os.getenv("GEMINI_TOP_P", "0.95"))
    max_output_tokens: int = int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "1024"))

    def require_api_key(self) -> str:
        """Return the API key or raise a clear error if it is missing."""
        if not self.api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Copy .env.example to .env and add "
                "your key (get one at https://aistudio.google.com/apikey)."
            )
        return self.api_key


SETTINGS = Settings()
