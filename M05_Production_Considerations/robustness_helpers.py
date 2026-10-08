"""Synthetic claim-form drawing for isolated robustness experiments."""
import platform
import textwrap
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

_FONTS = {  # (regular, bold) per operating system
    "Darwin": (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    ),
    "Windows": ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf"),
    "Linux": (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ),
}


def font(size, bold=False):
    path = _FONTS.get(platform.system(), ("", ""))[bold]
    if path and Path(path).exists():
        return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)  # Pillow >= 10.1


def draw_claim_form(path, invoice_id, patient_name, description, height=760):
    """Draw a Dentaire Plus claim form for CHF 800 with this description, save it to path, return the image."""
    regular, bold, bold_small = font(22), font(26, bold=True), font(18, bold=True)
    img = Image.new("RGB", (900, height), "white")
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 899, height - 1], outline=(200, 200, 200), width=2)
    d.text(
        (40, 40),
        "Groupe Mutuel \u2013 Reimbursement Claim Form",
        font=bold,
        fill=(20, 20, 20),
    )
    d.line([(40, 85), (860, 85)], fill=(180, 180, 180), width=2)
    y = 110
    for label, value in [
        ("Invoice / Claim ID:", invoice_id),
        ("Patient name:", patient_name),
        ("Date of birth:", "1990-01-01"),
        ("Policy holder number:", "GM-TEST-9999"),
        ("Insurance product:", "Dentaire Plus"),
        ("Policy start date:", "2020-01-01"),
        ("Provider:", "Cabinet Dentaire Rive Gauche"),
        ("Provider type:", "Dentist"),
        ("Treatment date:", "2025-12-01"),
    ]:
        d.text((40, y), label, font=bold_small, fill=(60, 60, 60))
        d.text((320, y), value, font=regular, fill=(0, 0, 0))
        y += 38
    d.text((40, y), "Treatment description:", font=bold_small, fill=(60, 60, 60))
    wrapped = "\n".join(textwrap.wrap(description, 58))
    d.multiline_text((320, y), wrapped, font=regular, fill=(0, 0, 0))
    y += 30 * (wrapped.count("\n") + 1) + 20
    d.text((40, y), "Total amount billed:", font=bold_small, fill=(60, 60, 60))
    d.text((320, y), "CHF 800.00", font=bold, fill=(0, 0, 0))
    img.save(path)
    return img
