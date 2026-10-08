"""Instructor checks for the core metrics; no API calls or notebook kernel required.

Run: uv run pytest M05_Production_Considerations/test_core_metrics.py
"""
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentic_workshop.invoices import load_ground_truth


@pytest.fixture(scope="module")
def metrics():
    notebook = json.loads(Path(__file__).with_name("M05_01_Evaluation.ipynb").read_text())
    namespace = {"load_ground_truth": load_ground_truth}
    assignments = {"CLAIM_IDS", "expected", "SCORE_AMOUNT"}
    functions = {"grade_amount", "is_correct", "repeated_metrics"}
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        for node in ast.parse("".join(cell["source"])).body:
            assign = isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id in assignments for target in node.targets
            )
            function = isinstance(node, ast.FunctionDef) and node.name in functions
            if assign or function:
                exec(compile(ast.Module(body=[node], type_ignores=[]), "core_metrics", "exec"), namespace)
    return namespace


def state(verdict, amount):
    return {"decision": SimpleNamespace(verdict=verdict, reimbursed_amount_chf=amount)}


@pytest.mark.parametrize("suffix,verdict,amount,correct", [
    ("003", "rejected", 0, True),
    ("003", "rejected", 0.01, True),
    ("003", "rejected", 0.02, False),
    ("003", "rejected", None, False),
    ("003", "approved", 0, False),
    ("003", "insufficient_information", None, False),
    ("011", "approved", None, True),
    ("011", "approved", 999, True),
    ("011", "rejected", None, False),
    ("012", "insufficient_information", None, True),
    ("012", "insufficient_information", 0, False),
    ("012", "approved", None, False),
])
def test_correctness_rules(metrics, suffix, verdict, amount, correct):
    assert metrics["is_correct"](f"INV-2025-{suffix}", state(verdict, amount)) is correct


@pytest.mark.parametrize("missing", [None, {"decision": None}])
@pytest.mark.parametrize("suffix", ["003", "011", "012"])
def test_missing_attempt_is_never_success(metrics, missing, suffix):
    assert metrics["is_correct"](f"INV-2025-{suffix}", missing) is False


@pytest.mark.parametrize("outcomes,any_success,all_success", [
    ([True, False, False], True, False),
    ([True, True, True], True, True),
    ([False, False, False], False, False),
    ([False, True, False], True, False),
])
def test_observed_reliability(metrics, outcomes, any_success, all_success):
    assert metrics["repeated_metrics"](outcomes) == {"Pass@k": any_success, "Pass^k": all_success}


def test_failed_attempts_remain_in_reliability_outcomes(metrics):
    attempts = [state("insufficient_information", None), None, {"decision": None}]
    outcomes = [metrics["is_correct"]("INV-2025-012", attempt) for attempt in attempts]
    assert outcomes == [True, False, False]
    assert metrics["repeated_metrics"](outcomes) == {"Pass@k": True, "Pass^k": False}


def test_empty_observation_is_not_perfect_reliability(metrics):
    with pytest.raises(ValueError):
        metrics["repeated_metrics"]([])
