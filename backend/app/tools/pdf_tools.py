"""PDF generation helpers for the report agent."""
import base64
import datetime
from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    Table,
    TableStyle,
)


def get_styles():
    styles = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "CustomTitle", parent=styles["Title"],
            fontSize=24, spaceAfter=12, textColor=colors.HexColor("#1a1a2e"),
        ),
        "h1": ParagraphStyle(
            "H1", parent=styles["Heading1"],
            fontSize=16, spaceBefore=16, spaceAfter=8,
            textColor=colors.HexColor("#16213e"),
        ),
        "h2": ParagraphStyle(
            "H2", parent=styles["Heading2"],
            fontSize=13, spaceBefore=12, spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "Body", parent=styles["Normal"],
            fontSize=10, spaceAfter=6, leading=14,
        ),
    }


def b64_to_image(b64_str: str, width: float = 6 * inch, height: float = 3.6 * inch) -> Image:
    img_data = base64.b64decode(b64_str)
    return Image(BytesIO(img_data), width=width, height=height)


def dark_table_style(col_widths=None) -> TableStyle:
    return TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("PADDING", (0, 0), (-1, -1), 6),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f9fa")]),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#dee2e6")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ])


def build_cover_table(state: dict) -> Table:
    best_score = max(
        (e.get("primary_score", 0) for e in state.get("experiments", [])
         if e.get("status") in ("completed", "completed_reduced")),
        default=0,
    )
    profile = state.get("profile") or {}
    cover_data = [
        ["Job ID", state.get("job_id", "")[:16] + "..."],
        ["Date", datetime.datetime.now().strftime("%Y-%m-%d %H:%M")],
        ["Goal", (state.get("user_description") or "")[:80]],
        ["Dataset", f"{profile.get('n_rows', 0):,} rows × {profile.get('n_cols', 0)} columns"],
        ["Best Model", state.get("best_model_name") or "N/A"],
        ["Primary Score", str(round(best_score, 4))],
    ]
    t = Table(cover_data, colWidths=[1.8 * inch, 4.5 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#2c3e50")),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.white),
        ("BACKGROUND", (1, 0), (1, -1), colors.HexColor("#ecf0f1")),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("PADDING", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#bdc3c7")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
    ]))
    return t
