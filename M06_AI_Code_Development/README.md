# M06 — AI for Code Development

In M04 Exercise 3, an invoice specialist and a policy specialist report to a supervisor. In this exercise, **your coding agent does the implementation**: move that setup into a Python function and expose it through a CLI. You guide the agent, run the tests and review its changes.

## Start here

Use the environment from the root README (`uv sync`). From the repository root:

```bash
cd M06_AI_Code_Development
uv run pytest
```

The starter has two unimplemented functions, so the contract tests initially fail. They run without an API key and block accidental API use. Your goal is to make them pass without weakening their assertions. Keep the working directory here when starting your coding agent.

## Exercise: specify and steer the coding agent

Your task is to **write your own prompt** for a coding agent to turn the M04 Exercise 3 claims team into a CLI. Decide what context the agent needs, explain the expected behavior, and guide it until the tests pass. Use the M04 notebook, the starter project and the tests to work out the requirements.

The finished application should accept an invoice ID, run the M04 claims team and print its structured decision as JSON. It should support help, a step limit and verbose tracing, and report failures clearly. The commands below and the tests define the expected interface.

1. **Inspect the starting point.** Run the tests and read the two starter functions. Revisit the M04 team you will transfer. Identify which existing code can be reused and which notebook state needs to become part of the application.
2. **Write your prompt.** Describe the goal, expected CLI behavior, relevant source files and acceptance criteria. Limit edits to this Module 06 project, require the existing tests to remain intact, and tell the agent not to read `.env` or make live API calls. 
3. **Review a plan.** Ask the agent to explain its approach before editing. Check that it preserves the invoice specialist, policy specialist and supervisor. Correct missing assumptions or unnecessary changes.
4. **Implement and iterate.** Let the agent write the code and run the offline tests. Use failures to refine your instructions. Review its changes rather than accepting a passing test report alone.
5. **Demonstrate and explain.** Run the finished CLI yourself, including one live claim and a missing invoice. Explain how the code handles decision state, repeated searches, output and errors.

Keep your initial prompt, one follow-up instruction and the final test results for discussion. Which requirement did your first prompt miss, and how did you correct it?

## Layout

```text
claims_pipeline/
  agent.py        TODO: turn the M04 team setup into run_agent(invoice_id)
  __main__.py     TODO: argument parsing, JSON output and exit codes
tests/
  conftest.py     prevents accidental API calls in tests
  test_agent.py   real shared agent loop with fake model replies and fake I/O
  test_cli.py     CLI contract, including a subprocess check of --help
```

Reusable code lives in `agentic_workshop/` at the repository root. The M04 notebook is the source of the team setup and prompts; the tests provide the interface contract.

## Run the finished CLI

```bash
uv run python -m claims_pipeline --help
uv run python -m claims_pipeline adjudicate INV-2025-003
uv run python -m claims_pipeline adjudicate INV-2025-003 --verbose --max-steps 15
uv run python -m claims_pipeline adjudicate INV-2025-999
```

The adjudication commands for existing invoices call the live API, and incur model and embedding costs. Normal output is one JSON decision; verbose traces go to stderr. The missing invoice should produce a concise error and exit code 1 before any API call. This exercise reports a proposed coverage decision; it does not pay a claim or implement a human approval workflow.

## Review and discuss

- Run `uv run pytest` and inspect the generated diff. Can you explain how the notebook's tools and decision state became functions?
- Verify that repeated policy searches share one index within a run, and that a later run cannot reuse a previous decision.
- Try one live claim yourself. Compare the specialist trace with M04, including citations and missing evidence.
- Explain why a passing offline test suite proves application behavior, while M05 evaluations are needed to assess decision quality.
