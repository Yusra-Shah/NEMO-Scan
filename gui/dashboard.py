"""
NEMO Scan - Dashboard Panel
Location: gui/dashboard.py

Doctor-centric workflow dashboard.
All data loaded from MongoDB via database/db.py.
"""

from datetime import datetime, timezone

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFrame, QScrollArea, QGridLayout,
    QDialog, QLineEdit, QTextEdit, QComboBox,
    QFormLayout, QMessageBox, QSizePolicy,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QCursor, QFont

import database.db as db

from gui.styles import (
    SURFACE, BORDER, TEXT, TEXT_SEC, BLUE, RED, GREEN, YELLOW,
    BLUE_TINT, RED_TINT, GREEN_TINT, YELLOW_TINT,
    BG, FONT_FAMILY, CONTENT_PAD, FONT_BODY, FONT_LABEL,
    FONT_METRIC, FONT_HEADER, CARD_RADIUS, BTN_RADIUS
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _label(text, size=12, color=TEXT, bold=False):
    l = QLabel(text)
    f = QFont(FONT_FAMILY, size)
    f.setBold(bold)
    l.setFont(f)
    l.setStyleSheet(f"color: {color}; background: transparent;")
    return l


def _pill_btn(text, filled=True, color=BLUE, height=40, min_width=0):
    btn = QPushButton(text)
    btn.setFixedHeight(height)
    if min_width:
        btn.setMinimumWidth(min_width)
    btn.setCursor(QCursor(Qt.PointingHandCursor))
    btn.setFont(QFont(FONT_FAMILY, 11))
    if filled:
        btn.setStyleSheet(f"""
            QPushButton {{
                background: {color}; color: white; border: none;
                border-radius: {height // 2}px; padding: 0 20px;
                font-weight: 600;
            }}
            QPushButton:hover {{ background: #3367D6; }}
        """)
    else:
        btn.setStyleSheet(f"""
            QPushButton {{
                background: {SURFACE}; color: {color};
                border: 1.5px solid {color};
                border-radius: {height // 2}px; padding: 0 20px;
                font-weight: 600;
            }}
            QPushButton:hover {{ background: {BLUE_TINT}; }}
        """)
    return btn


def _divider():
    f = QFrame()
    f.setFrameShape(QFrame.HLine)
    f.setStyleSheet(f"color: {BORDER}; background: {BORDER};")
    f.setFixedHeight(1)
    return f


def _fmt_time(dt) -> str:
    if not dt:
        return ""
    if isinstance(dt, str):
        return dt
    try:
        now = datetime.now(timezone.utc)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        diff = now - dt
        secs = int(diff.total_seconds())
        if secs < 60:
            return "just now"
        if secs < 3600:
            return f"{secs // 60}m ago"
        if secs < 86400:
            return f"{secs // 3600}h ago"
        return dt.strftime("%b %d")
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Metric Card
# ---------------------------------------------------------------------------

class MetricCard(QFrame):
    def __init__(self, title, value, subtitle, accent, tint, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        self.setMinimumHeight(120)
        self._accent = accent
        self._build(title, value, subtitle, accent, tint)

    def _build(self, title, value, subtitle, accent, tint):
        self.setStyleSheet(f"""
            QFrame#card {{
                background-color: {SURFACE};
                border: 1px solid {BORDER};
                border-radius: {CARD_RADIUS}px;
            }}
            QFrame#card:hover {{ border-color: {accent}; }}
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(6)

        top = QHBoxLayout()
        title_lbl = QLabel(title.upper())
        title_lbl.setStyleSheet(
            f"font-size: {FONT_LABEL}px; color: {TEXT_SEC}; "
            f"font-family: '{FONT_FAMILY}'; font-weight: 600; letter-spacing: 0.5px;"
        )
        dot = QLabel()
        dot.setFixedSize(10, 10)
        dot.setStyleSheet(f"background-color: {accent}; border-radius: 5px;")
        top.addWidget(title_lbl)
        top.addStretch()
        top.addWidget(dot)
        layout.addLayout(top)

        self.value_lbl = QLabel(str(value))
        self.value_lbl.setStyleSheet(
            f"font-size: {FONT_METRIC}px; font-weight: 700; color: {TEXT}; font-family: '{FONT_FAMILY}';"
        )
        layout.addWidget(self.value_lbl)

        sub_lbl = QLabel(subtitle)
        sub_lbl.setStyleSheet(
            f"font-size: {FONT_LABEL}px; color: {TEXT_SEC}; font-family: '{FONT_FAMILY}';"
        )
        layout.addWidget(sub_lbl)

    def set_value(self, v):
        self.value_lbl.setText(str(v))


# ---------------------------------------------------------------------------
# Activity Row
# ---------------------------------------------------------------------------

class ActivityRow(QFrame):
    clicked = Signal(dict)

    def __init__(self, scan: dict, parent=None):
        super().__init__(parent)
        self._scan = scan
        self.setFixedHeight(68)
        self.setCursor(QCursor(Qt.PointingHandCursor))

        diagnosis = scan.get("prediction", "Unknown")
        is_pneu = diagnosis.upper() == "PNEUMONIA"
        accent   = RED if is_pneu else GREEN
        badge_bg = RED_TINT if is_pneu else GREEN_TINT
        badge_fg = RED if is_pneu else GREEN

        self.setStyleSheet(f"""
            QFrame {{
                background-color: {SURFACE};
                border: none;
                border-left: 3px solid {accent};
            }}
            QFrame:hover {{ background-color: {BG}; }}
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 0, 16, 0)
        layout.setSpacing(12)

        info = QVBoxLayout()
        info.setSpacing(2)
        patient_name = scan.get("patient_name", scan.get("patient_id", "Unknown"))
        name_lbl = QLabel(patient_name)
        name_lbl.setStyleSheet(
            f"font-size: {FONT_BODY}px; font-weight: 600; color: {TEXT}; font-family: '{FONT_FAMILY}';"
        )
        time_lbl = QLabel(_fmt_time(scan.get("scan_date")))
        time_lbl.setStyleSheet(
            f"font-size: {FONT_LABEL}px; color: {TEXT_SEC}; font-family: '{FONT_FAMILY}';"
        )
        info.addWidget(name_lbl)
        info.addWidget(time_lbl)
        layout.addLayout(info)
        layout.addStretch()

        conf = scan.get("confidence", 0.0)
        conf_lbl = QLabel(f"{conf * 100:.1f}% conf.")
        conf_lbl.setStyleSheet(
            f"font-size: {FONT_LABEL}px; color: {TEXT_SEC}; font-family: '{FONT_FAMILY}';"
        )
        layout.addWidget(conf_lbl)
        layout.addSpacing(12)

        badge = QLabel(diagnosis.capitalize())
        badge.setFixedHeight(22)
        badge.setContentsMargins(10, 0, 10, 0)
        badge.setAlignment(Qt.AlignCenter)
        badge.setStyleSheet(
            f"background-color: {badge_bg}; color: {badge_fg}; border-radius: 11px; "
            f"font-size: {FONT_LABEL}px; font-weight: 600; font-family: '{FONT_FAMILY}'; padding: 0 4px;"
        )
        layout.addWidget(badge)

    def mousePressEvent(self, event):
        self.clicked.emit(self._scan)
        super().mousePressEvent(event)


# ---------------------------------------------------------------------------
# Register Patient Dialog
# ---------------------------------------------------------------------------

class RegisterPatientDialog(QDialog):
    patient_registered = Signal(dict)

    def __init__(self, doctor: dict, parent=None):
        super().__init__(parent)
        self.doctor = doctor
        self.setWindowTitle("Register New Patient")
        self.setFixedWidth(480)
        self.setStyleSheet(f"background: {BG};")
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 32, 32, 32)
        layout.setSpacing(16)

        layout.addWidget(_label("Register New Patient", 16, TEXT, bold=True))
        layout.addWidget(_label("Fill in patient details to create a record.", 11, TEXT_SEC))
        layout.addSpacing(8)

        def field(placeholder):
            f = QLineEdit()
            f.setPlaceholderText(placeholder)
            f.setFixedHeight(40)
            f.setFont(QFont(FONT_FAMILY, 11))
            f.setStyleSheet(f"""
                QLineEdit {{
                    background: {SURFACE}; border: 1.5px solid {BORDER};
                    border-radius: 8px; padding: 0 12px; color: {TEXT};
                }}
                QLineEdit:focus {{ border-color: {BLUE}; }}
            """)
            return f

        form = QFormLayout()
        form.setSpacing(10)
        form.setLabelAlignment(Qt.AlignRight)

        self.f_id     = field("Leave blank to auto-generate")
        self.f_name   = field("Full name")
        self.f_age    = field("Age")
        self.f_gender = QComboBox()
        self.f_gender.addItems(["Male", "Female", "Other"])
        self.f_gender.setFixedHeight(40)
        self.f_gender.setFont(QFont(FONT_FAMILY, 11))
        self.f_gender.setStyleSheet(f"""
            QComboBox {{
                background: {SURFACE}; border: 1.5px solid {BORDER};
                border-radius: 8px; padding: 0 12px; color: {TEXT};
            }}
            QComboBox:focus {{ border-color: {BLUE}; }}
            QComboBox::drop-down {{ border: none; width: 28px; }}
        """)
        self.f_contact  = field("Phone / email")
        self.f_symptoms = QTextEdit()
        self.f_symptoms.setPlaceholderText("Current symptoms (cough, fever, chest pain...)")
        self.f_symptoms.setFixedHeight(80)
        self.f_symptoms.setFont(QFont(FONT_FAMILY, 11))
        self.f_symptoms.setStyleSheet(f"""
            QTextEdit {{
                background: {SURFACE}; border: 1.5px solid {BORDER};
                border-radius: 8px; padding: 8px 12px; color: {TEXT};
            }}
            QTextEdit:focus {{ border-color: {BLUE}; }}
        """)
        self.f_history = QTextEdit()
        self.f_history.setPlaceholderText("Relevant medical history, allergies, medications...")
        self.f_history.setFixedHeight(80)
        self.f_history.setFont(QFont(FONT_FAMILY, 11))
        self.f_history.setStyleSheet(f"""
            QTextEdit {{
                background: {SURFACE}; border: 1.5px solid {BORDER};
                border-radius: 8px; padding: 8px 12px; color: {TEXT};
            }}
            QTextEdit:focus {{ border-color: {BLUE}; }}
        """)

        lbl_style = f"font-size: 11px; color: {TEXT_SEC}; font-family: '{FONT_FAMILY}';"
        for lbl_text, widget in [
            ("Patient ID", self.f_id),
            ("Full Name *", self.f_name),
            ("Age *", self.f_age),
            ("Gender *", self.f_gender),
            ("Contact", self.f_contact),
            ("Symptoms", self.f_symptoms),
            ("Medical History", self.f_history),
        ]:
            lbl = QLabel(lbl_text)
            lbl.setStyleSheet(lbl_style)
            form.addRow(lbl, widget)

        layout.addLayout(form)

        self.error_lbl = _label("", 11, RED)
        self.error_lbl.hide()
        layout.addWidget(self.error_lbl)

        btn_row = QHBoxLayout()
        cancel = _pill_btn("Cancel", filled=False)
        cancel.clicked.connect(self.reject)
        save = _pill_btn("Register Patient", color=GREEN)
        save.clicked.connect(self._save)
        btn_row.addWidget(cancel)
        btn_row.addStretch()
        btn_row.addWidget(save)
        layout.addLayout(btn_row)

    def _save(self):
        name    = self.f_name.text().strip()
        age_str = self.f_age.text().strip()
        contact = self.f_contact.text().strip()
        pid     = self.f_id.text().strip() or None

        self.error_lbl.hide()

        if not name:
            self.error_lbl.setText("Patient name is required.")
            self.error_lbl.show()
            return
        if not age_str.isdigit():
            self.error_lbl.setText("Age must be a number.")
            self.error_lbl.show()
            return

        try:
            patient = db.register_patient(
                name=name,
                age=int(age_str),
                gender=self.f_gender.currentText(),
                contact=contact,
                symptoms=self.f_symptoms.toPlainText().strip(),
                medical_history=self.f_history.toPlainText().strip(),
                assigned_doctor_id=self.doctor["doctor_id"],
                patient_id=pid,
            )
            self.patient_registered.emit(patient)
            self.accept()
        except ValueError as e:
            self.error_lbl.setText(str(e))
            self.error_lbl.show()
        except Exception as e:
            self.error_lbl.setText(f"Error: {e}")
            self.error_lbl.show()


# ---------------------------------------------------------------------------
# Dashboard Panel
# ---------------------------------------------------------------------------

class DashboardPanel(QWidget):
    new_scan_requested    = Signal()
    open_patient_profile  = Signal(dict)   # emits patient dict

    def __init__(self, doctor: dict, parent=None):
        super().__init__(parent)
        self.doctor = doctor
        self.setStyleSheet(f"background-color: {BG};")
        self._build_ui()
        self.refresh()

    def _build_ui(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("background: transparent; border: none;")

        content = QWidget()
        content.setStyleSheet(f"background: {BG};")
        layout = QVBoxLayout(content)
        layout.setContentsMargins(CONTENT_PAD, CONTENT_PAD, CONTENT_PAD, CONTENT_PAD)
        layout.setSpacing(0)

        # Header
        name = self.doctor.get("name", "Doctor")
        header_row = QHBoxLayout()
        header_block = QVBoxLayout()
        greeting = _label(f"Good day, Dr. {name}", 22, TEXT, bold=True)
        sub = _label("Here is your diagnostic overview for today.", 12, TEXT_SEC)
        header_block.addWidget(greeting)
        header_block.addSpacing(4)
        header_block.addWidget(sub)

        upload_btn = _pill_btn("New Scan", height=40)
        upload_btn.clicked.connect(self.new_scan_requested.emit)

        header_row.addLayout(header_block)
        header_row.addStretch()
        header_row.addWidget(upload_btn, alignment=Qt.AlignBottom)
        layout.addLayout(header_row)
        layout.addSpacing(32)

        # Metric cards
        grid = QGridLayout()
        grid.setSpacing(16)
        self.card_patients = MetricCard("Total Patients",   "0",     "Registered patients",        BLUE,   BLUE_TINT)
        self.card_today    = MetricCard("Scans Today",      "0",     "Since midnight",             YELLOW, YELLOW_TINT)
        self.card_pneu     = MetricCard("Pneumonia Rate",   "0%",    "Of all scans",               RED,    RED_TINT)
        self.card_reports  = MetricCard("Reports Generated","0",     "PDF reports created",        GREEN,  GREEN_TINT)
        grid.addWidget(self.card_patients, 0, 0)
        grid.addWidget(self.card_today,    0, 1)
        grid.addWidget(self.card_pneu,     0, 2)
        grid.addWidget(self.card_reports,  0, 3)
        layout.addLayout(grid)
        layout.addSpacing(32)

        # Quick actions
        act_lbl = _label("Quick Actions", 14, TEXT, bold=True)
        layout.addWidget(act_lbl)
        layout.addSpacing(12)

        acts_row = QHBoxLayout()
        acts_row.setSpacing(12)

        self.btn_new_patient = _pill_btn("Register New Patient", filled=True,  color=BLUE)
        self.btn_find_patient = _pill_btn("Find Patient",        filled=False, color=BLUE)
        self.btn_new_scan    = _pill_btn("New Scan",             filled=False, color=BLUE)
        self.btn_all_patients = _pill_btn("All Patients",        filled=False, color=BLUE)

        self.btn_new_patient.clicked.connect(self._open_register_patient)
        self.btn_new_scan.clicked.connect(self.new_scan_requested.emit)

        for b in [self.btn_new_patient, self.btn_find_patient, self.btn_new_scan, self.btn_all_patients]:
            acts_row.addWidget(b)
        acts_row.addStretch()
        layout.addLayout(acts_row)
        layout.addSpacing(32)

        # Recent activity
        sec_lbl = _label("Recent Scans", 14, TEXT, bold=True)
        layout.addWidget(sec_lbl)
        layout.addSpacing(12)

        self.activity_frame = QFrame()
        self.activity_frame.setStyleSheet(
            f"background: {SURFACE}; border: 1px solid {BORDER}; border-radius: {CARD_RADIUS}px;"
        )
        self.activity_layout = QVBoxLayout(self.activity_frame)
        self.activity_layout.setContentsMargins(0, 0, 0, 0)
        self.activity_layout.setSpacing(0)

        self.empty_label = _label("No scans yet. Upload an X-ray to get started.", 12, TEXT_SEC)
        self.empty_label.setAlignment(Qt.AlignCenter)
        self.empty_label.setContentsMargins(0, 48, 0, 48)
        self.activity_layout.addWidget(self.empty_label)

        layout.addWidget(self.activity_frame)
        layout.addStretch()

        scroll.setWidget(content)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def refresh(self):
        """Reload all data from MongoDB."""
        try:
            stats = db.get_dashboard_stats(self.doctor["doctor_id"])
            self.card_patients.set_value(str(stats.get("total_patients", 0)))
            self.card_today.set_value(str(stats.get("scans_today", 0)))
            rate = stats.get("pneumonia_rate", 0.0)
            self.card_pneu.set_value(f"{rate}%")
            self.card_reports.set_value(str(stats.get("reports_generated", 0)))

            self._load_activity()
        except Exception as e:
            print(f"Dashboard refresh error: {e}")

    def _load_activity(self):
        # Clear old rows
        while self.activity_layout.count():
            item = self.activity_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        try:
            recent = db.get_recent_activity(self.doctor["doctor_id"], limit=5)
        except Exception:
            recent = []

        if not recent:
            self.empty_label = _label("No scans yet. Upload an X-ray to get started.", 12, TEXT_SEC)
            self.empty_label.setAlignment(Qt.AlignCenter)
            self.empty_label.setContentsMargins(0, 48, 0, 48)
            self.activity_layout.addWidget(self.empty_label)
            return

        for i, scan in enumerate(recent):
            row = ActivityRow(scan)
            row.clicked.connect(self._on_activity_clicked)
            self.activity_layout.addWidget(row)
            if i < len(recent) - 1:
                self.activity_layout.addWidget(_divider())

    def _on_activity_clicked(self, scan: dict):
        patient_id = scan.get("patient_id")
        if patient_id:
            patient = db.get_patient_by_id(patient_id)
            if patient:
                self.open_patient_profile.emit(patient)

    def _open_register_patient(self):
        dlg = RegisterPatientDialog(self.doctor, self)
        dlg.patient_registered.connect(self._on_patient_registered)
        dlg.exec()

    def _on_patient_registered(self, patient: dict):
        self.refresh()
        QMessageBox.information(
            self,
            "Patient Registered",
            f"Patient '{patient['name']}' registered successfully.\nID: {patient['patient_id']}",
        )