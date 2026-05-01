"""
PneumoScan - PDF Report Generator
core/report_generator.py

Generates an English A4 PDF diagnostic report using ReportLab built-in fonts.

Public API:
    generate_report(patient, doctor, result, model_votes, scan_date,
                    heatmap_path, doctor_notes, output_dir) -> str
    Returns the absolute path to the saved PDF.
"""

import os
import re
from datetime import datetime, timezone

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import (
    BaseDocTemplate, PageTemplate, Frame,
    Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Image, KeepTogether,
)
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.pdfgen import canvas as rl_canvas

# ---------------------------------------------------------------------------
# Brand palette (matches gui/styles.py)
# ---------------------------------------------------------------------------
C_BLUE        = colors.HexColor("#4285F4")
C_RED         = colors.HexColor("#EA4335")
C_GREEN       = colors.HexColor("#34A853")
C_YELLOW      = colors.HexColor("#FBBC05")
C_DARK        = colors.HexColor("#202124")
C_SEC         = colors.HexColor("#5F6368")
C_BG          = colors.HexColor("#F8F9FA")
C_BORDER      = colors.HexColor("#E8EAED")
C_RED_TINT    = colors.HexColor("#FCE8E6")
C_GREEN_TINT  = colors.HexColor("#E6F4EA")
C_BLUE_TINT   = colors.HexColor("#E8F0FE")

# ---------------------------------------------------------------------------
# Model display names
# ---------------------------------------------------------------------------
MODEL_DISPLAY = {
    "densenet121":     "DenseNet-121",
    "resnet50":        "ResNet-50",
    "efficientnet_b4": "EfficientNet-B4",
    "vit_b16":         "ViT-B/16",
    "mobilenetv3":     "MobileNetV3",
    "inception_v3":    "Inception-V3",
    "attention_cnn":   "Attention CNN",
}

# ---------------------------------------------------------------------------
# Style factory
# ---------------------------------------------------------------------------

def _styles():
    base = {
        "normal":     ParagraphStyle("normal",     fontName="Helvetica",         fontSize=9,  leading=13, textColor=C_DARK),
        "small":      ParagraphStyle("small",      fontName="Helvetica",         fontSize=8,  leading=11, textColor=C_SEC),
        "bold":       ParagraphStyle("bold",       fontName="Helvetica-Bold",    fontSize=9,  leading=13, textColor=C_DARK),
        "label":      ParagraphStyle("label",      fontName="Helvetica-Bold",    fontSize=7,  leading=10, textColor=C_SEC),
        "section":    ParagraphStyle("section",    fontName="Helvetica-Bold",    fontSize=10, leading=14, textColor=C_BLUE),
        "header":     ParagraphStyle("header",     fontName="Helvetica-Bold",    fontSize=18, leading=22, textColor=C_DARK),
        "subheader":  ParagraphStyle("subheader",  fontName="Helvetica",         fontSize=10, leading=14, textColor=C_SEC),
        "diag_normal":ParagraphStyle("diag_normal",fontName="Helvetica-Bold",    fontSize=22, leading=28, textColor=C_GREEN,  alignment=TA_CENTER),
        "diag_pneu":  ParagraphStyle("diag_pneu",  fontName="Helvetica-Bold",    fontSize=22, leading=28, textColor=C_RED,    alignment=TA_CENTER),
        "center":     ParagraphStyle("center",     fontName="Helvetica",         fontSize=9,  leading=13, textColor=C_DARK,   alignment=TA_CENTER),
        "bold_center":ParagraphStyle("bold_center",fontName="Helvetica-Bold",    fontSize=9,  leading=13, textColor=C_DARK,   alignment=TA_CENTER),
        "footer":     ParagraphStyle("footer",     fontName="Helvetica-Oblique", fontSize=7,  leading=10, textColor=C_SEC,    alignment=TA_CENTER),
        "notes":      ParagraphStyle("notes",      fontName="Helvetica",         fontSize=9,  leading=14, textColor=C_DARK),
    }
    return base


# ---------------------------------------------------------------------------
# Page decorator (header bar + footer on every page)
# ---------------------------------------------------------------------------

def _page_decorator(canvas, doc):
    canvas.saveState()
    w, h = A4

    # Top accent bar (Google-color stripe)
    stripe_colors = [C_BLUE, C_RED, C_YELLOW, C_GREEN]
    stripe_w = w / 4
    for i, sc in enumerate(stripe_colors):
        canvas.setFillColor(sc)
        canvas.rect(i * stripe_w, h - 6 * mm, stripe_w, 3 * mm, fill=1, stroke=0)

    # Header background
    canvas.setFillColor(C_BG)
    canvas.rect(0, h - 22 * mm, w, 16 * mm, fill=1, stroke=0)

    # Logo text: "Pneumo" in brand colors + " Scan"
    canvas.setFont("Helvetica-Bold", 15)
    x = 15 * mm
    y = h - 15 * mm
    for ch, clr in zip("Pneumo", [C_BLUE, C_RED, C_YELLOW, C_GREEN, C_BLUE, C_RED]):
        canvas.setFillColor(clr)
        canvas.drawString(x, y, ch)
        x += canvas.stringWidth(ch, "Helvetica-Bold", 15)
    canvas.setFillColor(C_DARK)
    canvas.drawString(x, y, " Scan")

    # Subtitle
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(C_SEC)
    canvas.drawString(15 * mm, h - 19 * mm, "AI-Powered Lung Diagnostic System  |  Prototype  |  Academic Use Only")

    # Report label (right-aligned)
    canvas.setFont("Helvetica-Bold", 7)
    canvas.setFillColor(C_SEC)
    canvas.drawRightString(w - 15 * mm, h - 12 * mm, "LUNG DIAGNOSIS REPORT")
    canvas.setFont("Helvetica", 7)
    canvas.drawRightString(w - 15 * mm, h - 17 * mm,
                           f"Page {doc.page}  |  Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

    # Bottom divider + footer
    canvas.setStrokeColor(C_BORDER)
    canvas.setLineWidth(0.5)
    canvas.line(15 * mm, 12 * mm, w - 15 * mm, 12 * mm)
    canvas.setFont("Helvetica-Oblique", 7)
    canvas.setFillColor(C_SEC)
    canvas.drawCentredString(
        w / 2, 7 * mm,
        "This report is for professional medical reference only.  "
        "Final diagnosis must be confirmed by a qualified physician."
    )

    canvas.restoreState()


# ---------------------------------------------------------------------------
# Section heading helper
# ---------------------------------------------------------------------------

def _section(title, styles):
    return KeepTogether([
        Paragraph(title.upper(), styles["section"]),
        HRFlowable(width="100%", thickness=0.5, color=C_BORDER, spaceAfter=4),
    ])


# ---------------------------------------------------------------------------
# Two-column info table helper
# ---------------------------------------------------------------------------

def _info_table(rows, col_widths=(55 * mm, 95 * mm)):
    style = TableStyle([
        ("FONT",        (0, 0), (0, -1), "Helvetica-Bold", 8),
        ("FONT",        (1, 0), (1, -1), "Helvetica",      9),
        ("TEXTCOLOR",   (0, 0), (0, -1), C_SEC),
        ("TEXTCOLOR",   (1, 0), (1, -1), C_DARK),
        ("TOPPADDING",  (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 3),
        ("VALIGN",      (0, 0), (-1, -1), "TOP"),
    ])
    return Table(rows, colWidths=col_widths, style=style)


# ---------------------------------------------------------------------------
# Diagnosis result box
# ---------------------------------------------------------------------------

def _diagnosis_box(prediction, confidence, ensemble_prob, severity, subtype, styles, page_w):
    is_pneu = "PNEUMONIA" in prediction.upper()
    accent  = C_RED   if is_pneu else C_GREEN
    bg      = C_RED_TINT if is_pneu else C_GREEN_TINT
    diag_st = "diag_pneu" if is_pneu else "diag_normal"
    verdict = prediction.upper()

    inner_w = page_w  # will be constrained by frame margins

    data = [
        [Paragraph(verdict, styles[diag_st])],
    ]
    box = Table(data, colWidths=[inner_w])
    box.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, -1), bg),
        ("BOX",          (0, 0), (-1, -1), 1.5, accent),
        ("ROUNDEDCORNERS", [8]),
        ("TOPPADDING",   (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 10),
    ]))

    # Metrics row beneath
    metrics_data = [
        [
            Paragraph("CONFIDENCE", styles["label"]),
            Paragraph("ENSEMBLE PROBABILITY", styles["label"]),
            Paragraph("SEVERITY", styles["label"]),
            Paragraph("SUBTYPE", styles["label"]),
        ],
        [
            Paragraph(f"{confidence * 100:.1f}%", styles["bold_center"]),
            Paragraph(f"{ensemble_prob * 100:.1f}%", styles["bold_center"]),
            Paragraph(str(severity) or "N/A", styles["bold_center"]),
            Paragraph(str(subtype)  or "N/A", styles["bold_center"]),
        ],
    ]
    col_w = inner_w / 4
    metrics = Table(metrics_data, colWidths=[col_w] * 4)
    metrics.setStyle(TableStyle([
        ("ALIGN",       (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
        ("FONT",        (0, 0), (-1, 0),  "Helvetica-Bold", 7),
        ("FONT",        (0, 1), (-1, 1),  "Helvetica-Bold", 11),
        ("TEXTCOLOR",   (0, 0), (-1, 0),  C_SEC),
        ("TEXTCOLOR",   (0, 1), (-1, 1),  accent),
        ("TOPPADDING",  (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 5),
        ("LINEBELOW",   (0, 0), (-1, 0),  0.3, C_BORDER),
        ("BOX",         (0, 0), (-1, -1), 0.5, C_BORDER),
        ("BACKGROUND",  (0, 0), (-1, -1), C_BG),
    ]))

    return KeepTogether([box, Spacer(1, 3 * mm), metrics])


# ---------------------------------------------------------------------------
# Model votes table
# ---------------------------------------------------------------------------

def _model_votes_table(model_votes, styles, page_w):
    header = [
        Paragraph("MODEL", styles["label"]),
        Paragraph("PNEUMONIA PROB.", styles["label"]),
        Paragraph("VERDICT", styles["label"]),
    ]
    rows = [header]
    col_widths = [70 * mm, 70 * mm, page_w - 140 * mm]

    for key in ["densenet121", "resnet50", "efficientnet_b4", "vit_b16",
                "mobilenetv3", "inception_v3", "attention_cnn"]:
        prob     = float(model_votes.get(key, 0.0))
        name     = MODEL_DISPLAY.get(key, key)
        verdict  = "Pneumonia" if prob >= 0.5 else "Normal"
        v_color  = C_RED if prob >= 0.5 else C_GREEN

        rows.append([
            Paragraph(name, styles["normal"]),
            Paragraph(f"{prob * 100:.1f}%", styles["bold"]),
            Paragraph(verdict, ParagraphStyle(
                f"v_{key}", fontName="Helvetica-Bold", fontSize=9,
                leading=13, textColor=v_color
            )),
        ])

    tbl = Table(rows, colWidths=col_widths, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0),  C_BLUE_TINT),
        ("FONT",         (0, 0), (-1, 0),  "Helvetica-Bold", 7),
        ("TEXTCOLOR",    (0, 0), (-1, 0),  C_BLUE),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, C_BG]),
        ("GRID",         (0, 0), (-1, -1), 0.4, C_BORDER),
        ("TOPPADDING",   (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 5),
        ("LEFTPADDING",  (0, 0), (-1, -1), 8),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return tbl


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_report(
    patient: dict,
    doctor: dict,
    result: dict,
    model_votes: dict,
    scan_date: datetime,
    heatmap_path: str = "",
    doctor_notes: str = "",
    output_dir: str = "outputs/reports",
) -> str:
    """
    Build the PDF and save it.  Returns the absolute path to the file.

    result dict keys used:
        prediction, confidence, ensemble_prob, severity, subtype
    model_votes keys:
        densenet121, resnet50, efficientnet_b4, vit_b16,
        mobilenetv3, inception_v3, attention_cnn
    """
    os.makedirs(output_dir, exist_ok=True)

    patient_id        = patient.get("patient_id", "UNKNOWN")
    patient_name_safe = re.sub(r'[^A-Za-z0-9]', '_', patient.get("name", "Unknown").replace(' ', '_'))
    ts                = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename          = f"report_{patient_id}_{patient_name_safe}_{ts}.pdf"
    filepath   = os.path.join(output_dir, filename)

    prediction    = result.get("prediction", "Unknown")
    confidence    = float(result.get("confidence", 0.0))
    ensemble_prob = float(result.get("ensemble_prob", 0.0))
    severity      = result.get("severity", "None") or "None"
    subtype       = result.get("subtype",  "N/A")  or "N/A"

    page_w, page_h = A4
    margin         = 15 * mm
    usable_w       = page_w - 2 * margin

    doc = BaseDocTemplate(
        filepath,
        pagesize=A4,
        leftMargin=margin,
        rightMargin=margin,
        topMargin=28 * mm,
        bottomMargin=18 * mm,
        title=f"PneumoScan Report - {patient_id}",
        author="PneumoScan AI Diagnostic System",
    )
    frame = Frame(
        margin, 18 * mm,
        usable_w, page_h - 46 * mm,
        id="body",
    )
    template = PageTemplate(id="main", frames=[frame], onPage=_page_decorator)
    doc.addPageTemplates([template])

    styles = _styles()
    story  = []

    # ------------------------------------------------------------------
    # Report title block
    # ------------------------------------------------------------------
    scan_date_str = (
        scan_date.strftime("%B %d, %Y  %H:%M UTC")
        if scan_date else datetime.now(timezone.utc).strftime("%B %d, %Y  %H:%M UTC")
    )

    title_data = [[
        Paragraph("LUNG DIAGNOSIS REPORT", styles["header"]),
        Paragraph(
            f"Report Date: {scan_date_str}<br/>Patient ID: {patient_id}",
            ParagraphStyle("rt", fontName="Helvetica", fontSize=8, leading=12,
                           textColor=C_SEC, alignment=TA_RIGHT),
        ),
    ]]
    title_tbl = Table(title_data, colWidths=[usable_w * 0.6, usable_w * 0.4])
    title_tbl.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "BOTTOM"),
        ("TOPPADDING",   (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 0),
    ]))
    story.append(title_tbl)
    story.append(Spacer(1, 5 * mm))
    story.append(HRFlowable(width="100%", thickness=1.5, color=C_BLUE, spaceAfter=5))

    # ------------------------------------------------------------------
    # Patient & Doctor info — flat rows, no nested KeepTogether
    # ------------------------------------------------------------------
    info_label_w = 35 * mm
    info_val_w   = usable_w / 2 - info_label_w - 8 * mm
    gap_col      = 8 * mm

    def _cell_lbl(t):
        return Paragraph(t, ParagraphStyle(
            "il", fontName="Helvetica-Bold", fontSize=8, leading=12, textColor=C_SEC))

    def _cell_val(t):
        return Paragraph(str(t), ParagraphStyle(
            "iv", fontName="Helvetica", fontSize=9, leading=12, textColor=C_DARK))

    info_rows_data = [
        # header row
        [
            Paragraph("PATIENT INFORMATION", ParagraphStyle(
                "ph", fontName="Helvetica-Bold", fontSize=9, leading=12, textColor=C_BLUE)),
            Paragraph(""),
            Paragraph(""),
            Paragraph("PHYSICIAN", ParagraphStyle(
                "dh", fontName="Helvetica-Bold", fontSize=9, leading=12, textColor=C_BLUE)),
            Paragraph(""),
            Paragraph(""),
        ],
        [_cell_lbl("Name"),         _cell_val(patient.get("name", "-")),          Paragraph(""),
         _cell_lbl("Doctor"),       _cell_val(f"Dr. {doctor.get('name', '-')}"),  Paragraph("")],
        [_cell_lbl("Patient ID"),   _cell_val(patient.get("patient_id", "-")),    Paragraph(""),
         _cell_lbl("Specialization"), _cell_val(doctor.get("specialization", "-")), Paragraph("")],
        [_cell_lbl("Age"),          _cell_val(str(patient.get("age", "-"))),      Paragraph(""),
         _cell_lbl("Scan Date"),    _cell_val(scan_date_str),                    Paragraph("")],
        [_cell_lbl("Gender"),       _cell_val(patient.get("gender", "-")),        Paragraph(""),
         Paragraph(""),             Paragraph(""),                                Paragraph("")],
        [_cell_lbl("Contact"),      _cell_val(patient.get("contact", "-") or "-"), Paragraph(""),
         Paragraph(""),             Paragraph(""),                                Paragraph("")],
    ]

    cw = [info_label_w, info_val_w, gap_col, info_label_w, info_val_w, gap_col]
    info_tbl = Table(info_rows_data, colWidths=cw)
    info_tbl.setStyle(TableStyle([
        ("SPAN",        (0, 0), (2, 0)),   # patient header spans 3 cols
        ("SPAN",        (3, 0), (5, 0)),   # doctor header spans 3 cols
        ("LINEBELOW",   (0, 0), (2, 0),  0.5, C_BLUE),
        ("LINEBELOW",   (3, 0), (5, 0),  0.5, C_BLUE),
        ("TOPPADDING",  (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 3),
        ("VALIGN",      (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 2),
    ]))
    story.append(info_tbl)
    story.append(Spacer(1, 6 * mm))

    # ------------------------------------------------------------------
    # Diagnosis result
    # ------------------------------------------------------------------
    story.append(_section("Diagnostic Result", styles))
    story.append(_diagnosis_box(prediction, confidence, ensemble_prob,
                                severity, subtype, styles, usable_w))
    story.append(Spacer(1, 6 * mm))

    # ------------------------------------------------------------------
    # Heatmap (if available)
    # ------------------------------------------------------------------
    if heatmap_path and os.path.isfile(heatmap_path):
        story.append(_section("Grad-CAM Heatmap", styles))
        story.append(Paragraph(
            "The heatmap highlights regions of the X-ray that most influenced the diagnosis.",
            styles["small"],
        ))
        story.append(Spacer(1, 2 * mm))
        try:
            max_w = usable_w * 0.55
            max_h = 65 * mm
            img = Image(heatmap_path, width=max_w, height=max_h, kind="proportional")
            story.append(img)
        except Exception:
            story.append(Paragraph("(Heatmap image could not be loaded)", styles["small"]))
        story.append(Spacer(1, 6 * mm))

    # ------------------------------------------------------------------
    # Model votes
    # ------------------------------------------------------------------
    story.append(_section("AI Model Ensemble Votes  (7 Models)", styles))
    story.append(_model_votes_table(model_votes, styles, usable_w))
    story.append(Spacer(1, 6 * mm))

    # ------------------------------------------------------------------
    # Doctor notes
    # ------------------------------------------------------------------
    story.append(_section("Doctor Notes", styles))
    notes_text = doctor_notes.strip() if doctor_notes else "(No notes recorded)"
    notes_data = [[Paragraph(notes_text, styles["notes"])]]
    notes_tbl  = Table(notes_data, colWidths=[usable_w])
    notes_tbl.setStyle(TableStyle([
        ("BOX",          (0, 0), (-1, -1), 0.5, C_BORDER),
        ("BACKGROUND",   (0, 0), (-1, -1), C_BG),
        ("TOPPADDING",   (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 8),
        ("LEFTPADDING",  (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(notes_tbl)

    doc.build(story)
    return os.path.abspath(filepath)
