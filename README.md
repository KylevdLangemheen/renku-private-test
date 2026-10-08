# FCUE LLM Agentic Workshop — Hands-on

Throughout the day we build on one running example: medical invoices (images of Groupe Mutuel claim forms) are extracted into structured data, checked against the insurance terms and conditions with RAG, adjudicated by an agent, evaluated, and finally moved into a CLI with a coding agent.

## Setup

1. Install [uv](https://docs.astral.sh/uv/), then, from the repo root:

   ```bash
   uv sync
   ```

   This creates `.venv/` with every dependency and the shared `agentic_workshop` package.
2. In the hosted workspace, `init.d/setup_project.sh` adds a command to `~/.bashrc`
   that loads `OPENAI_API_KEY` from `/api-token/api_token.txt` when the file is
   readable and no key is already set. New Bash terminals, including the pi
   agent, inherit the key. In an already-open terminal, run `source ~/.bashrc`
   before starting pi or Jupyter. If Jupyter is already running without the key,
   restart it from that terminal so its kernels inherit the environment.
   For local setup without the token mount, create an ignored root `.env` file
   containing `OPENAI_API_KEY=your-key`; notebooks load it automatically.
   Never put the key in a notebook cell or in public environment settings.
3. Start Jupyter from the repo root and select the `.venv` kernel:

   ```bash
   uv run jupyter lab
   ```

Models: `gpt-6-luna` (text, vision, tools) and `text-embedding-3-small` (embeddings), set in `agentic_workshop/config.py`.

## Modules

| Time | Module | Hands-on |
| --- | --- | --- |
| 9:15–9:40 | M01 Foundations | `M01_Foundations/M01_01_Foundations_Demo.ipynb` (instructor demo) |
| 9:40–10:55 | M02 Structured Output & Tool Calling | `M02_Structured_Output_Tool_Calling/M02_01_Structured_Output.ipynb`, `M02_02_Tool_Calling.ipynb` |
| 11:10–12:20 | M03 Retrieval-Augmented Generation | `M03_Retrieval_Augmented_Generation/M03_01_RAG_Base.ipynb` to `M03_04_RAG_Visual.ipynb` |
| 13:10–14:40 | M04 Agentic Patterns | `M04_Agentic_Patterns/M04_01_Agentic_Patterns.ipynb` |
| 14:55–16:05 | M05 Production Considerations | `M05_Production_Considerations/M05_01_Evaluation.ipynb` (core: correctness and reliability); optional: `M05_02_Component_Evaluation.ipynb`, `M05_03_Prompt_Injection.ipynb` |
| 16:05–16:50 | M06 AI for Code Development | `M06_AI_Code_Development/` (port the M04 decision agent to a CLI with offline tests) |

M06 asks students to use a coding agent to port the M04 Exercise 3 claims team into a CLI. Its starter tests intentionally fail until students implement the agent adapter and command-line interface. Run them from its project folder:

```bash
cd M06_AI_Code_Development
uv run pytest
```

## Repository layout

```
agentic_workshop/   code shared by every module: paths, client, models, prices, invoices,
                    policy documents, MiniVectorStore, RAG helpers
data/
  invoices/           12 synthetic claim forms + hand-checked ground truth
  insurance_terms/    6 public Groupe Mutuel terms & conditions PDFs
  insurance_figures/  4 synthetic figures + questions, for M03.04
  generate_data.ipynb how the synthetic data was made (instructor prep)
M0x_*/              one folder per module; module-local helpers sit next to the notebooks
                    (M05 runner and bonus helpers)
```

## Conventions for authors

- Notebooks hold the **complete** solution. Code marked `# TODO (Exercise N): ...` is removed when the notebook is handed to students.
- `# SUGGESTION: ...` marks a demo that may no longer show its point with the current model.
- Notebooks take paths, the client and model names from `agentic_workshop`, never from the working directory: Jupyter starts each kernel in the notebook's own folder.
