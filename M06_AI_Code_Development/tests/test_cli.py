"""Public CLI contract; the student implements main, not these assertions."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from agentic_workshop.claims_agent import SubmitDecisionArgs
from M06_AI_Code_Development.claims_pipeline import __main__ as cli


@pytest.fixture
def decision():
    return SubmitDecisionArgs(
        verdict="rejected", reimbursed_amount_chf=0,
        rationale="Cosmetic treatment is excluded.", citations=["CAGA04-E2 Art. 18"],
    )


def test_adjudicate_prints_one_json_decision(monkeypatch, capsys, decision):
    calls = []

    def fake_run(invoice_id, *, max_steps, verbose):
        calls.append((invoice_id, max_steps, verbose))
        return decision

    monkeypatch.setattr(cli, "run_agent", fake_run)
    assert cli.main(["adjudicate", "INV-2025-003"]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out) == decision.model_dump()
    assert captured.err == ""
    assert calls == [("INV-2025-003", 15, False)]


def test_options_reach_agent_and_verbose_trace_stays_off_stdout(monkeypatch, capsys, decision):
    calls = []

    def fake_run(invoice_id, *, max_steps, verbose):
        calls.append((invoice_id, max_steps, verbose))
        print("TRACE: search_policy")  # shared M04 loop prints its trace to stdout
        return decision

    monkeypatch.setattr(cli, "run_agent", fake_run)
    assert cli.main(["adjudicate", "INV-2025-003", "--max-steps", "4", "--verbose"]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out) == decision.model_dump()
    assert "TRACE: search_policy" in captured.err
    assert calls == [("INV-2025-003", 4, True)]


@pytest.mark.parametrize("outcome", [None, FileNotFoundError("No invoice INV-2025-999"), RuntimeError("API unavailable")])
def test_failures_return_one_with_diagnostic_and_no_json(monkeypatch, capsys, outcome):
    def fake_run(*args, **kwargs):
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(cli, "run_agent", fake_run)
    assert cli.main(["adjudicate", "INV-2025-999"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.strip()
    assert "Traceback" not in captured.err


@pytest.mark.parametrize("argv", [[], ["unknown"], ["adjudicate"], ["adjudicate", "INV-2025-003", "--max-steps", "0"], ["adjudicate", "INV-2025-003", "--max-steps", "-1"]])
def test_invalid_arguments_exit_two_before_agent_runs(monkeypatch, argv):
    def unexpected(*args, **kwargs):
        pytest.fail("Argument errors must not start the agent.")

    monkeypatch.setattr(cli, "run_agent", unexpected)
    with pytest.raises(SystemExit) as error:
        cli.main(argv)
    assert error.value.code == 2


def test_module_help_runs_in_a_real_process_without_api_key():
    project = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env.pop("OPENAI_API_KEY", None)
    # Mirror pytest's pythonpath in a subprocess, even when pytest starts at repo root.
    env["PYTHONPATH"] = os.pathsep.join([str(project), str(project.parents[1])])
    result = subprocess.run(
        [sys.executable, "-m", "claims_pipeline", "--help"], cwd=project,
        env=env, capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert "adjudicate" in result.stdout
    assert "Traceback" not in result.stderr
