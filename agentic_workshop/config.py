"""Paths, models and the OpenAI client shared by every module.

The paths are anchored to this package, not to the working directory: Jupyter starts each kernel in
the notebook's own folder, so a relative "data/..." would point somewhere different in every module.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
INVOICES_DIR = DATA_DIR / "invoices"
TERMS_DIR = DATA_DIR / "insurance_terms"
GENERATED_DIR = DATA_DIR / "generated"  # files written by the notebooks (git-ignored)

CHAT_MODEL = "gpt-6-luna"  # text, vision and tool calling
OLD_MODEL = "gpt-4o-mini"  # Older text + vision model to showcase failure modes
EMBEDDING_MODEL = "text-embedding-3-small"

# Reasoning configs for simplicity
REASONING_OFF = {"effort": "none"}
REASONING_DEFAULT = None
