import functools

from dotenv import load_dotenv
from openai import OpenAI

from .config import REPO_ROOT


@functools.cache
def get_client() -> OpenAI:
    """The OpenAI client, created and cached on first use with the key from <repo>/.env (OPENAI_API_KEY)."""
    load_dotenv(REPO_ROOT / ".env")
    return OpenAI()
