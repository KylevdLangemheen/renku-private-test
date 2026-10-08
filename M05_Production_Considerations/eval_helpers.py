"""Run and instrument the existing Module 4 claims team.

Metrics stay in notebooks. Usage includes invoice extraction and nested specialist /
supervisor responses; embedding calls are excluded. Failed extraction attempts
retain a None usage entry even when the specialist catches the error. try_run_agent preserves its
None-on-failure interface, so failed runs can lose partial usage records.
"""
import time
from pathlib import Path
import openai
from pydantic import BaseModel, Field
from agentic_workshop.agents import Tool
from agentic_workshop.claims_agent import SubmitDecisionArgs, build_claims_team, run_supervisor
from agentic_workshop.client import get_client
from agentic_workshop.config import INVOICES_DIR
from agentic_workshop.invoices import extract_invoice
from agentic_workshop.rag import search_policy

class ReadInvoiceArgs(BaseModel):
    invoice_id: str = Field(description="The claim ID, e.g. INV-2025-001.")


class RetrievePolicyClauseArgs(BaseModel):
    query: str = Field(
        description="What to look up in the policy documents, as a specific question."
    )


def run_agent(
    invoice_id, system_prompt, model, store, max_steps=15, verbose=False,
    invoice_image_path=None,
):
    """Run the Module 4 multi-agent claims system on one invoice ID.

    model: chat model. store: the structured policy store the agent searches.

    Returns a state dict with the extracted invoice (if available), the decision (None if the
    supervisor did not submit one), token usage for every model call, retrieved chunks, retrieval
    and tool-call counts, and latency.
    ``invoice_image_path`` is only used for generated adversarial forms; ordinary runs resolve
    the ID under data/invoices/.
    """
    state = {
        "invoice_id": invoice_id,
        "image_path": invoice_image_path or INVOICES_DIR / f"{invoice_id}.png",
        "usage": [],
        "model_calls": 0,
        "tool_trace": [],
        "retrieval_calls": 0,
        "tool_calls": 0,
        "retrieved_doc_codes": [],
        "retrieved_chunks": [],
        "decision": None,
    }

    def read_invoice_tool(invoice_id: str):
        if invoice_id != state["invoice_id"]:
            raise ValueError(f"This run is for {state['invoice_id']}, not {invoice_id}.")
        if not Path(state["image_path"]).exists():
            raise ValueError(f"No invoice {invoice_id}.")
        state["model_calls"] += 1
        try:
            invoice, usage = extract_invoice(state["image_path"], model)
        except (openai.OpenAIError, ValueError):
            # The invoice specialist may catch this and continue: retain the usage gap.
            state["usage"].append(None)
            raise
        state["usage"].append(usage)
        if invoice is None:
            raise ValueError("Invoice extraction returned no structured result.")
        state["invoice"] = invoice
        return invoice.model_dump()

    def retrieve_policy_tool(query: str):
        if "invoice" not in state:
            raise ValueError("Ask the invoice specialist to read the invoice first.")
        invoice = state["invoice"]
        state["retrieval_calls"] += 1
        hits = search_policy(
            store,
            invoice.insurance_product,
            invoice.treatment_date,
            query,
        )
        state["retrieved_doc_codes"].extend(hit["source"] for hit in hits)
        state["retrieved_chunks"].extend(hits)
        return hits

    def submit_decision_tool(verdict, reimbursed_amount_chf=None, rationale="", citations=None):
        decision = SubmitDecisionArgs(
            verdict=verdict,
            reimbursed_amount_chf=reimbursed_amount_chf,
            rationale=rationale,
            citations=citations or [],
        )
        state["decision"] = decision
        return {"received": True}

    invoice_tool = Tool.from_pydantic(
        ReadInvoiceArgs,
        read_invoice_tool,
        name="read_invoice",
        description="Read the claim form for the requested invoice ID and return its structured fields.",
    )
    policy_tool = Tool.from_pydantic(
        RetrievePolicyClauseArgs,
        retrieve_policy_tool,
        name="search_policy",
        description="Search three citable extracts from the policy in force for the invoice's product and treatment date (taken from the invoice automatically). Search again with a more specific query when needed.",
    )
    submit_tool = Tool.from_pydantic(
        SubmitDecisionArgs,
        submit_decision_tool,
        name="submit_decision",
        description="Submit the final structured coverage decision. Call exactly once when ready.",
    )

    def record_response(response):
        state["usage"].append(response.usage)
        state["model_calls"] += 1

    def record_tool_call(tool_call):
        state["tool_calls"] += 1
        state["tool_trace"].append({"name": tool_call.name, "arguments": tool_call.arguments})

    team = build_claims_team(
        client=get_client(),
        model=model,
        invoice_tool=invoice_tool,
        policy_search_tool=policy_tool,
        submit_decision_tool=submit_tool,
        system_prompt=system_prompt,
        max_steps=max_steps,
        verbose=verbose,
        on_response=record_response,
        on_tool_call=record_tool_call,
    )

    def decision_submitted(name, _result):
        return name == "submit_decision"

    start = time.perf_counter()
    run_supervisor(
        team,
        f"Adjudicate claim {invoice_id}. Read this invoice, investigate its coverage, and submit a decision.",
        stop_when=decision_submitted,
    )
    state["latency_s"] = time.perf_counter() - start
    return state


def try_run_agent(invoice_id, system_prompt, *args, **kwargs):
    """Like run_agent, but report API failures and return None so other runs can continue.

    A completed run with no decision is a state dict whose ``decision`` value is None.
    """
    try:
        return run_agent(invoice_id, system_prompt, *args, **kwargs)
    except (openai.OpenAIError, ValueError) as e:
        print(
            f"{invoice_id}: failed ({type(e).__name__}), counted as no decision"
            + (f": {e}" if isinstance(e, ValueError) else "")
        )
        return None
