"""Student adapter: port M04 Exercise 3 into a callable Python module."""

from agentic_workshop.claims_agent import SubmitDecisionArgs

def run_agent(
    invoice_id: str, *, max_steps: int = 15, verbose: bool = False
) -> SubmitDecisionArgs | None:
    """Run the M04 claims team and return its submitted decision, or None.

    TODO: move notebook setup into this function, with fresh decision state per call.
    Validate the invoice path before initializing the client or building the index.
    Build the index once per run, reuse it across searches, and stop on submit_decision.
    See TASK_BRIEF.md and tests/test_agent.py for the contract.
    """
    raise NotImplementedError("Port the M04 decision agent here with your coding agent.")
