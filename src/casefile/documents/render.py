"""Render document contents to PDF through Jinja2 templates.

The same content always produces the same bytes: the creation date written
into every PDF is fixed instead of being the time of the run.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal

from fpdf import FPDF
from jinja2 import Environment, PackageLoader, StrictUndefined
from pydantic import BaseModel

from casefile.documents.content import (
    INSURER_NAME,
    CaseDocuments,
    ClaimFormContent,
    PoliceReportContent,
    RepairEstimateContent,
)

FIXED_CREATION_DATE = datetime(2015, 3, 1, tzinfo=UTC)

CLAIM_FORM_FILE = "claim_form.pdf"
REPAIR_ESTIMATE_FILE = "repair_estimate.pdf"
POLICE_REPORT_FILE = "police_report.pdf"

_TEMPLATES: dict[type[BaseModel], tuple[str, str]] = {
    ClaimFormContent: ("claim_form.html.j2", "Automobile Claim Form"),
    RepairEstimateContent: ("repair_estimate.html.j2", "Vehicle Repair Estimate"),
    PoliceReportContent: ("police_report.html.j2", "Incident Report"),
}

_environment = Environment(
    loader=PackageLoader("casefile.documents"),
    autoescape=True,
    undefined=StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
)
# Each document type writes dates its own way, as real paperwork does. All
# three formats name the month or put the year first, so none is ambiguous.
_environment.filters["money"] = lambda amount: f"{Decimal(amount):,.2f}"
_environment.filters["long_date"] = lambda day: f"{day:%B} {day.day}, {day.year}"
_environment.filters["short_date"] = lambda day: f"{day:%b} {day.day}, {day.year}"
_environment.filters["iso_date"] = date.isoformat


def render_pdf(content: BaseModel) -> bytes:
    """Render one document to PDF bytes."""
    template_name, title = _TEMPLATES[type(content)]
    html = _environment.get_template(template_name).render(
        doc=content, insurer=INSURER_NAME
    )
    pdf = FPDF(format="Letter")
    pdf.set_creation_date(FIXED_CREATION_DATE)
    pdf.set_title(title)
    pdf.add_page()
    pdf.set_font("Helvetica", size=10)
    pdf.write_html(html)
    return bytes(pdf.output())


def render_case(documents: CaseDocuments) -> dict[str, bytes]:
    """Render every document of a case, keyed by file name."""
    files = {CLAIM_FORM_FILE: render_pdf(documents.claim_form)}
    if documents.repair_estimate is not None:
        files[REPAIR_ESTIMATE_FILE] = render_pdf(documents.repair_estimate)
    if documents.police_report is not None:
        files[POLICE_REPORT_FILE] = render_pdf(documents.police_report)
    return files
