# Task: port the M04 decision agent to a CLI

## Source and scope

Port **Exercise 3: Build a multi-agent claims adjudicator** from `M04_Agentic_Patterns/M04_01_Agentic_Patterns.ipynb`. Preserve the invoice specialist, policy specialist and supervisor that submits a structured decision. Move notebook setup into ordinary Python; reuse the shared agent loop and team builder.

Read these files (paths relative to the repository root):

- `M04_Agentic_Patterns/M04_01_Agentic_Patterns.ipynb`: Exercise 3 setup, tool definitions and supervisor prompt, plus the preceding index setup.
- `agentic_workshop/claims_agent.py`: `SubmitDecisionArgs`, `build_claims_team`, `run_supervisor`.
- `agentic_workshop/agents.py`: `Tool`, `ToolCallingAgent`, `stop_when` and verbose tracing.
- `agentic_workshop/invoices.py`: `extract_invoice` returns `(MedicalInvoice, usage)`.
- `agentic_workshop/rag.py`: `load_documents`, `split_structured`, `build_store`, `search_policy`.
- `agentic_workshop/config.py` and `documents.py`: repository paths, model and policy metadata.
- This project's starter files and tests.

Keep edits inside this Module 06 project. Do not change M04, shared helpers or the existing test assertions. Do not read `.env`, embed keys in source, or launch live API calls through the coding agent. The student runs the final live check.

## Deliverable 1: callable decision agent

Implement the existing signature in `claims_pipeline/agent.py`:

```python
run_agent(invoice_id: str, *, max_steps: int = 15, verbose: bool = False) -> SubmitDecisionArgs | None
```

- Accept an invoice ID, not an arbitrary path. Reject inputs containing path separators. Resolve `INVOICES_DIR / f"{invoice_id}.png"` and check it exists **before** getting the client, loading PDFs or building the index. Missing invoices raise `FileNotFoundError`; path inputs may raise `ValueError`.
- Initialize the client and policy store inside the function. Importing the module must not call the API. Build the index once per run using `load_documents()`, `split_structured`, and `metadata=lambda code: POLICY_DOCUMENTS[code]`.
- Define the M04 `read_invoice` tool: call `extract_invoice(path, model=CHAT_MODEL)`, unpack the tuple and return the invoice's `model_dump()`.
- Define the M04 `search_policy` tool: accept `product`, `treatment_date` and `query`, and pass them to the shared `search_policy(store, product, treatment_date, query)`. Reuse the store across searches. Preserve policy edition filtering through the shared helper.
- Define `submit_decision` with the M04 schema. Its function validates arguments with the shared `SubmitDecisionArgs` and records the typed result in **local state**, fresh for each run.
- Use `Tool` and `build_claims_team` with those three tools, `CHAT_MODEL`, the M04 supervisor instructions, `max_steps` and `verbose`. Preserve specialist isolation and repeated policy searches. The limit applies to each agent's loop, not to the total number of calls across the team.
- Call `run_supervisor` with a task naming the invoice ID and `stop_when=lambda name, _result: name == "submit_decision"`. Return the recorded decision; return `None` if no decision was submitted, including when the loop reaches its limit. Never substitute a guessed decision.

The starter imports shared dependencies at module level so tests can replace the external I/O there. Keep these patch points; you may add private helpers or argument models.

## Deliverable 2: command-line interface

Implement `main(argv: list[str] | None = None) -> int` in `claims_pipeline/__main__.py` with `argparse`. Keep the entry point `sys.exit(main())`.

```bash
uv run python -m claims_pipeline adjudicate INV-2025-003 [--max-steps 15] [--verbose]
```

| Case | Output | Exit code |
| --- | --- | --- |
| Decision submitted | Exactly one JSON object on stdout, using the decision's fields | 0 |
| Missing invoice, runtime/API error, or no submitted decision | Concise diagnostic on stderr, empty stdout, no traceback | 1 |
| Missing/invalid command or arguments; nonpositive step limit | argparse usage/error, before starting the agent | 2 |
| `--help` | Help text listing `adjudicate`; no client initialization or index building | 0 |

The options belong to the `adjudicate` subcommand. Default to 15 steps and quiet mode. Pass both options explicitly to `run_agent`. In verbose mode, send the shared agent's printed traces to stderr (for example with `contextlib.redirect_stdout`) so stdout remains valid JSON. Do not prompt for approval or add an evaluation command in the core task.

## Tests and workflow

1. Read the source and tests, ask the coding agent for a short plan, and run the baseline `uv run pytest`.
2. Have the coding agent implement the port. Re-run the suite and use failures to refine the implementation without weakening assertions.
3. Review the diff and ask the agent to explain how state, tools, output and errors are handled.
4. Run `--help`, then run one existing invoice yourself with `--verbose` and the missing `INV-2025-999`. Check stdout, stderr and exit status (`echo $?` immediately after a command).

The provided tests replace model replies, extraction and retrieval with fakes, while exercising the real shared team and tool loop. They check typed decisions, repeated searches, specialist tool access, per-run state, step limits, early file validation, CLI options, output and failures. They do **not** prove that live decisions are correct, that the model will always call tools in this sequence, or that prompts resist every attack. Use M05's evaluation approach for those questions.

If you add a feature, add a relevant test rather than changing the existing contract. Optional stretch: write decisions to a file with an `--output` option, or add a subprocess test using an explicitly injected offline runner. Human review can be a later extension after the port works.
