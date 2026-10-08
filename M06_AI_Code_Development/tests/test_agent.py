"""Exercise the real M04 team with scripted model replies and fake I/O."""
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agentic_workshop.claims_agent import SubmitDecisionArgs
from M06_AI_Code_Development.claims_pipeline import agent

INVOICE_ID = "INV-2025-003"
INVOICE = {
    "invoice_id": INVOICE_ID, "insurance_product": "Dentaire Plus",
    "treatment_date": "2025-05-15", "treatment_description": "Teeth whitening",
    "amount_chf": 800,
}
DECISION = {
    "verdict": "rejected", "reimbursed_amount_chf": 0,
    "rationale": "Cosmetic treatment is excluded.", "citations": ["CAGA04-E2 Art. 18"],
}


def reply(name=None, arguments=None, text=""):
    output = [] if name is None else [SimpleNamespace(
        type="function_call", name=name, arguments=json.dumps(arguments), call_id=name,
    )]
    return SimpleNamespace(output=output, output_text=text)


@pytest.fixture
def dependencies(monkeypatch, tmp_path):
    (tmp_path / f"{INVOICE_ID}.png").write_bytes(b"fake image; extraction is mocked")
    monkeypatch.setattr(agent, "INVOICES_DIR", tmp_path)
    store = object()
    documents = {"CAGA04-E2": "fake policy"}
    mocks = {
        "get_client": Mock(),
        "load_documents": Mock(return_value=documents),
        "build_store": Mock(return_value=store),
        "extract_invoice": Mock(return_value=(SimpleNamespace(model_dump=lambda: INVOICE), None)),
        "search_policy": Mock(return_value=[{
            "source": "CAGA04-E2", "article": "Art. 18", "text": "Cosmetic exclusion",
        }]),
    }
    for name, mock in mocks.items():
        monkeypatch.setattr(agent, name, mock)
    return tmp_path, store, documents, mocks


def test_team_reads_searches_repeatedly_and_submits_typed_decision(dependencies):
    folder, store, documents, mocks = dependencies
    observed_tools = []
    responses = iter([
        reply("ask_invoice_agent", {"task": f"Read {INVOICE_ID}"}),
        reply("read_invoice", {"invoice_id": INVOICE_ID}),
        reply(text=json.dumps(INVOICE)),
        reply("ask_policy_agent", {"task": json.dumps(INVOICE)}),
        reply("search_policy", {"product": "Dentaire Plus", "treatment_date": "2025-05-15", "query": "benefits"}),
        reply("search_policy", {"product": "Dentaire Plus", "treatment_date": "2025-05-15", "query": "cosmetic exclusions"}),
        reply(text="CAGA04-E2 Art. 18 excludes cosmetic treatment."),
        reply("submit_decision", DECISION),
    ])

    def create(**kwargs):
        observed_tools.append({tool["name"] for tool in kwargs["tools"]})
        assert kwargs["model"] == agent.CHAT_MODEL
        return next(responses)  # another call after submit_decision fails the test

    client = SimpleNamespace(responses=SimpleNamespace(create=create))
    mocks["get_client"].return_value = client
    decision = agent.run_agent(INVOICE_ID, max_steps=4, verbose=False)
    assert isinstance(decision, SubmitDecisionArgs)
    assert decision.model_dump() == DECISION
    mocks["extract_invoice"].assert_called_once_with(folder / f"{INVOICE_ID}.png", model=agent.CHAT_MODEL)
    mocks["load_documents"].assert_called_once_with()
    mocks["build_store"].assert_called_once()
    args, kwargs = mocks["build_store"].call_args
    assert args == (documents, agent.split_structured)
    assert kwargs["metadata"]("CAGA04-E2") == agent.POLICY_DOCUMENTS["CAGA04-E2"]
    assert mocks["search_policy"].call_count == 2
    assert [call.args for call in mocks["search_policy"].call_args_list] == [
        (store, "Dentaire Plus", "2025-05-15", "benefits"),
        (store, "Dentaire Plus", "2025-05-15", "cosmetic exclusions"),
    ]
    assert observed_tools == [
        {"ask_invoice_agent", "ask_policy_agent", "submit_decision"},
        {"read_invoice"}, {"read_invoice"},
        {"ask_invoice_agent", "ask_policy_agent", "submit_decision"},
        {"search_policy"}, {"search_policy"}, {"search_policy"},
        {"ask_invoice_agent", "ask_policy_agent", "submit_decision"},
    ]


def test_no_submitted_decision_does_not_reuse_previous_run(dependencies):
    mocks = dependencies[3]
    create = Mock(side_effect=[reply("submit_decision", DECISION), reply(text="Unable to decide")])
    mocks["get_client"].return_value = SimpleNamespace(responses=SimpleNamespace(create=create))
    assert agent.run_agent(INVOICE_ID).model_dump() == DECISION
    assert agent.run_agent(INVOICE_ID) is None


def test_step_limit_returns_none(dependencies):
    mocks = dependencies[3]
    create = Mock(return_value=reply("ask_invoice_agent", {"task": "Read invoice"}))
    mocks["get_client"].return_value = SimpleNamespace(responses=SimpleNamespace(create=create))
    assert agent.run_agent(INVOICE_ID, max_steps=1) is None
    assert create.call_count == 2  # one supervisor step and one specialist step


@pytest.mark.parametrize("invoice_id", ["INV-2025-999", "../outside", "/tmp/outside"])
def test_missing_or_path_input_fails_before_api_or_index(dependencies, invoice_id):
    mocks = dependencies[3]
    with pytest.raises((ValueError, FileNotFoundError)):
        agent.run_agent(invoice_id)
    mocks["get_client"].assert_not_called()
    mocks["load_documents"].assert_not_called()
    mocks["build_store"].assert_not_called()
