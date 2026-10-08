"""Student CLI: uv run python -m claims_pipeline adjudicate INV-2025-003"""

import argparse
import sys

from .agent import run_agent


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, call the agent, print JSON, and return an exit code.

    TODO: implement the CLI contract in TASK_BRIEF.md.
    Accept argv so tests can call this without changing the process arguments.
    """
    raise NotImplementedError("Build the command-line interface with your coding agent.")


if __name__ == "__main__":
    sys.exit(main())
