"""The medical invoices in data/invoices/: images, image schema and their ground truth."""

import base64
import json
from pathlib import Path

from pydantic import BaseModel, Field

from .client import get_client
from .config import CHAT_MODEL, INVOICES_DIR


class MedicalInvoice(BaseModel):
    invoice_id: str
    patient_name: str
    patient_date_of_birth: str = Field(description="As YYYY-MM-DD.")
    policy_holder_number: str
    insurance_product: str
    policy_start_date: str = Field(description="As YYYY-MM-DD.")
    provider_name: str
    provider_type: str
    treatment_date: str = Field(description="As YYYY-MM-DD.")
    treatment_description: str
    notes: str | None = None
    amount_chf: float | None = Field(
        description="Use null if any digit is not clearly legible."
    )


def invoice_paths() -> list[Path]:
    """The 12 claim forms, INV-2025-001.png to INV-2025-012.png."""
    return sorted(INVOICES_DIR.glob("INV-2025-*.png"))


def load_ground_truth(invoice_id: str) -> dict:
    """The hand-checked labels of one invoice: {"extraction": {...}, "coverage_decision": {...}}."""
    # encoding="utf-8" matters: without it, Windows reads "Léa" as "LÃ©a"
    path = INVOICES_DIR / "ground_truth" / f"{invoice_id}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def image_to_data_url(path: Path) -> str:
    """A PNG file as a data URL, the form the API accepts for input images."""
    return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode()


def extract_invoice(image_path: Path, model: str = CHAT_MODEL):
    """Read a claim-form image. Returns (MedicalInvoice, usage); unreadable fields are null."""
    response = get_client().responses.parse(
        model=model,
        input=[
            {
                "role": "system",
                "content": (
                    "Extract the fields from this Groupe Mutuel reimbursement claim form. "
                    "If a field is illegible or missing, use null rather than guessing."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Extract the invoice fields."},
                    {"type": "input_image", "image_url": image_to_data_url(image_path)},
                ],
            },
        ],
        text_format=MedicalInvoice,
    )
    return response.output_parsed, response.usage
