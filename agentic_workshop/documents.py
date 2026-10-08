"""The Groupe Mutuel terms & conditions PDFs in data/insurance_terms/."""

import functools
import re

from pypdf import PdfReader

from .config import TERMS_DIR

GENERAL_CONDITIONS = "General Conditions (CGC)"

# One entry per PDF. Several editions of one product can exist: only the one in force on the
# treatment date applies, together with the General Conditions.
POLICY_DOCUMENTS = {
    "CAGA04-E2": {"filename": "CAGA04-E2.pdf", "product": GENERAL_CONDITIONS, "effective_date": "2022-01-01"},
    "DPAM01-E4": {"filename": "DPAM01-E4_portefeuille_ferme.pdf", "product": "Dentaire Plus", "effective_date": "2010-09-01"},
    "SOGA01-E1": {"filename": "SOGA01-E1.pdf", "product": "Optimum", "effective_date": "2023-09-01"},
    "RFGM02-E3": {"filename": "RFGM02-E3.pdf", "product": "PrimaCare", "effective_date": "2024-01-01"},
    "RFBA02-E5": {"filename": "RFBA02-E5.pdf", "product": "PrimaCare", "effective_date": "2025-04-01"},
    "RSGM02-E5": {"filename": "RSGM02-E5.pdf", "product": "OptiMed", "effective_date": "2025-04-01"},
}


@functools.cache
def load_pdf_text(doc_code: str, strip_toc: bool = True) -> str:
    """Text of one policy PDF (read once, then cached).

    Each document opens with a table of contents that lists "Art. 1 ...", "Art. 2 ...". With
    strip_toc, it is cut out: the body starts at the second "Art. 1".
    """
    reader = PdfReader(TERMS_DIR / POLICY_DOCUMENTS[doc_code]["filename"])
    text = "\n".join(page.extract_text() for page in reader.pages)
    if strip_toc:
        starts = list(re.finditer(r"(?m)^Art\.\s*1[ \t]", text))
        if len(starts) >= 2:
            text = text[:starts[0].start()] + text[starts[1].start():]
    return text


def editions_in_force(treatment_date: str) -> set[str]:
    """The doc_code of each product's latest edition with effective_date <= treatment_date (YYYY-MM-DD)."""
    latest = {}  # product -> doc_code
    for doc_code, doc in POLICY_DOCUMENTS.items():
        if doc["effective_date"] > treatment_date:
            continue
        current = latest.get(doc["product"])
        if current is None or doc["effective_date"] > POLICY_DOCUMENTS[current]["effective_date"]:
            latest[doc["product"]] = doc_code
    return set(latest.values())


def applicable_documents(insurance_product: str, treatment_date: str) -> set[str]:
    """The documents that apply to a claim: the insured's product and the General Conditions, in force on the treatment date."""
    return {doc_code for doc_code in editions_in_force(treatment_date)
            if POLICY_DOCUMENTS[doc_code]["product"] in {insurance_product, GENERAL_CONDITIONS}}
