"""Comparison-index plumbing for the optional component experiments."""
import re

import pandas as pd

from agentic_workshop.documents import load_pdf_text
from agentic_workshop.invoices import invoice_paths, load_ground_truth
from agentic_workshop.rag import ARTICLE_HEADING, build_store, search_policy

def build_comparison_store(doc_metadata, splitter):
    """Build the comparison index for the chunking exercise; splitter maps text to chunks."""
    documents = {code: load_pdf_text(code) for code in doc_metadata}
    def comparison_metadata(code):
        return doc_metadata[code]

    return build_store(documents, splitter, metadata=comparison_metadata)


def cited_articles(citations, citation_re):
    """{(doc_code, article number)} named in citations, found with citation_re:
    ["DPAM01-E4 Art.3.1"] -> {("DPAM01-E4", "3")}."""
    return {m.groups() for c in citations if (m := citation_re.search(c))}


def chunk_article(chunk):
    """(source, article number) a retrieved chunk comes from: the article it starts in, i.e. its own
    heading if it begins with one, else the last heading before it in the document; None before Art. 1."""
    own_heading = ARTICLE_HEADING.match(chunk["text"])
    if own_heading:  # also covers pieces whose heading was repeated, which is not verbatim in the document
        return chunk["source"], re.match(r"Art\. (\d+)", own_heading.group()).group(1)
    document = load_pdf_text(chunk["source"])
    start = document.find(chunk["text"])
    headings = [m for m in ARTICLE_HEADING.finditer(document) if m.start() <= start]
    return (chunk["source"], re.match(r"Art\. (\d+)", headings[-1].group()).group(1)) if headings else None


def is_citable(chunk):
    """True if the chunk's text starts with an article heading, i.e. the model can see which article it is reading."""
    return ARTICLE_HEADING.match(chunk["text"]) is not None


def show_chunks_containing(doc_code, text, splitter):
    """Print every chunk of doc_code that contains text: its header line, its length, and its start."""
    for chunk in splitter(load_pdf_text(doc_code)):
        if text in chunk:
            print(
                f"--- chunk headed {chunk.splitlines()[0]!r} ({len(chunk)} characters) ---"
            )
            print(chunk[:600], "\n")


def load_retrieval_cases(citation_re):
    """(invoice_id, extracted fields, relevant {(doc_code, article)}) for each invoice whose
    ground-truth decision cites an article."""
    cases = []
    for path in invoice_paths():
        gt = load_ground_truth(path.stem)
        relevant = cited_articles(gt["coverage_decision"]["citations"], citation_re)
        if relevant:  # INV-2025-012 has no article to find
            cases.append((path.stem, gt["extraction"], relevant))
    return cases


def show_retrieved(store, inv, relevant, query, k=5):
    """Print the top-k chunks for one query: rank, hit (✓/✗), credited article, start of the text."""
    for rank, hit in enumerate(search_policy(store, inv["insurance_product"], inv["treatment_date"], query, k=k), 1):
        article = chunk_article(hit)
        label = f"{article[0]} Art. {article[1]}" if article else "no article"
        print(f"{rank}. {'✓' if article in relevant else '✗'} {label:<20} {' '.join(hit['text'].split())[:70]}...")
    print("Relevant:", ", ".join(f"{doc} Art. {art}" for doc, art in sorted(relevant)))


def grade_amount(actual, expected, tolerance=0.01):
    """Same grading rule as M05.01: numeric amounts match within tolerance; null matches only null."""
    if expected is None:
        return actual is None
    return actual is not None and abs(actual - expected) <= tolerance + 1e-9


def grade_run(invoice_id, state):
    """Score one run with the M05.01 rules: missing runs fail; 011 is verdict-only; 012 requires a null amount.

    `review` flags the runs a human should look at: no decision, an abstention, or no payout amount.
    """
    expected = load_ground_truth(invoice_id)["coverage_decision"]
    decision = state.get("decision") if state else None
    verdict_ok = decision is not None and decision.verdict == expected["verdict"]
    amount_scored = invoice_id != "INV-2025-011"
    amount_ok = ((grade_amount(decision.reimbursed_amount_chf, expected["reimbursed_amount_chf"])
                  if decision is not None else False) if amount_scored else None)
    success = bool(verdict_ok and (amount_ok if amount_scored else True))
    if decision is None:
        outcome = "failed run" if state is None else "missing decision"
    elif decision.verdict == "insufficient_information":
        outcome = ("expected abstention" if expected["verdict"] == decision.verdict
                   else "unexpected abstention")
    else:
        outcome = "correct decision" if success else "wrong decision"
    return {"invoice_id": invoice_id, "verdict_correct": verdict_ok,
            "amount_scored": amount_scored, "amount_correct": amount_ok,
            "task_success": success, "outcome": outcome,
            "review": (decision is None or decision.verdict == "insufficient_information"
                       or decision.reimbursed_amount_chf is None)}


def resource_row(state):
    """Model calls, searches, tool calls and latency of one run (None for a failed run)."""
    return {"model_calls": state["model_calls"] if state else None,
            "searches": state["retrieval_calls"] if state else None,
            "tool_calls": state["tool_calls"] if state else None,
            "latency_s": state["latency_s"] if state else None}


def results_table(states):
    """One row per run: the grade of `grade_run` and the resources of `resource_row`."""
    return pd.DataFrame([{**grade_run(i, s), **resource_row(s)}
                         for i, s in states.items()]).set_index("invoice_id")
