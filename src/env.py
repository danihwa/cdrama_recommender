from pathlib import Path
from dotenv import load_dotenv

_SECRETS = Path.home() / "secrets" / ".env"


def load_secrets() -> None:
    """Load env vars from ~/secrets/.env if present, else walk up from current working directory."""
    if _SECRETS.exists():
        load_dotenv(_SECRETS)
    else:
        load_dotenv()