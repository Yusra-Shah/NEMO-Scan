"""
NEMO Scan - Patient Records Panel
Location: gui/patients_panel.py

Loads real patient data from MongoDB.
Supports search, patient profile view, and full scan history per patient.
"""

from datetime import datetime, timezone

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QTableWidget, QTableWidgetItem, QHeaderView, QLineEdit,
    QPushButton, QScrollArea, QSizePolicy, QStackedWidget,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QCursor, QFont

import database.db as db

from gui.styles import (
    SURFACE, BORDER, TEXT, TEXT_SEC, BLUE, RED, GREEN, YELLOW,
    BLUE_TINT, RED_TINT, GREEN_TINT, BG, FONT_FAMILY, CONTENT_PAD,
    FONT_BODY, FONT_LABEL, FONT_HEADER, CARD_RADIUS, BTN_RADIUS
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


def _pill_btn(text, filled=True, color=BLUE, height=36):
    btn = QPushButton(text)
    btn.setFixedHeight(height)
    btn.setCursor(QCursor(Qt.PointingHandCursor))
    btn.setFont(QFont(FONT_FAMILY, 11))
    if filled:
        btn.setStyleSheet(f"""
            QPushButton {{
                background: {color}; color: white; border: none;
                border-radius: {height // 2}px; padding: 0 18px; font-weight: 600;
            }}
            QPushButton:hover {{ background: #3367D6; }}
        """)
    else:
        btn.setStyleSheet(f"""
            QPushButton {{
                background: {SURFACE}; color: {color};
                border: 1.5px solid {color};
                border-radius: {height // 2}px; padding: 0 18px; font-weight: 600;
            }}
            QPushButton:hover {{ background: {BLUE_TINT}; }}
        """)
    return btn


def _fmt_date(dt) -> str:
    if not dt:
        return "Never"
    if isinstance(dt, str):
        return dt
    try:
        return dt.strftime("%b %d, %Y")
    except Exception:
        return str(dt)


def _divider():
    f = QFrame()
    f.setFrameShape(QFrame.HLine)
    f.setStyleSheet(f"background: {BORDER};")
    f.setFixedHeight(1)
    return f


# ---------------------------------------------------------------------------
# Scan History Row (inside patient profile)
# ---------------------------------------------------------------------------

class ScanHistoryRow(QFrame):
    view_requested = Signal(dict)

    def __init__(self, scan: dict, parent=None):
        super().__init__(parent)
        self._scan = scan
        self.setFixedHeight(64)
        self.setCursor(QCursor(Qt.PointingHandCursor))

        prediction = scan.get("result", {}).get("prediction", "Unknown")
        confidence = scan.get("result", {}).get("confidence", 0.0)
        severity   = scan.get("result", {}).get("severity", "")
        subtype    = scan.get("result", {}).get("subtype", "")
        scan_date  = scan.get("scan_date")

        is_pneu  = prediction.upper() == "PNEUMONIA"
        accent   = RED if is_pneu else GREEN
        badge_bg = RED_TINT if is_pneu else GREEN_TINT
        badge_fg = RED if is_pneu else GREEN

        self.setStyleSheet(f"""
            QFrame {{
                background: {SURFACE};
                border: none;
                border-left: 3px solid {accent};
            }}
            QFrame:hover {{ background: {BG}; }}
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 0, 16, 0)
        layout.setSpacing(12)

        info = QVBoxLayout()
        info.setSpacing(2)
        date_lbl = _label(_fmt_date(scan_date), FONT_BODY, TEXT, bold=True)
        detail = f"{severity}  |  {subtype}" if severity and subtype else severity or subtype or ""
        detail_lbl = _label(detail, 10, TEXT_SEC)
        info.addWidget(date_lbl)
        info.addWidget(detail_lbl)
        layout.addLayout(info)
        layout.addStretch()

        conf_lbl = _label(f"{confidence * 100:.1f}%", FONT_LABEL, TEXT_SEC)
        layout.addWidget(conf_lbl)
        layout.addSpacing(12)

        badge = QLabel(prediction.capitalize())
        badge.setFixedHeight(22)
        badge.setContentsMargins(10, 0, 10, 0)
        badge.setAlignment(Qt.AlignCenter)
        badge.setStyleSheet(
            f"background: {badge_bg}; color: {badge_fg}; border-radius: 11px; "
            f"font-size: {FONT_LABEL}px; font-weight: 600; font-family: '{FONT_FAMILY}';"
        )
        layout.addWidget(badge)

    def mousePressEvent(self, event):
        self.view_requested.emit(self._scan)
        super().mousePressEvent(event)


# ---------------------------------------------------------------------------
# Patient Profile Panel
# ---------------------------------------------------------------------------

class PatientProfilePanel(QWidget):
    back_requested = Signal()
    new_scan_requested = Signal(dict)   # emits patient dict

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background: {BG};")
        self._patient = None
        self._build_ui()

    def _build_ui(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("background: transparent; border: none;")

        self._content = QWidget()
        self._content.setStyleSheet(f"background: {BG};")
        self._main_layout = QVBoxLayout(self._content)
        self._main_layout.setContentsMargins(CONTENT_PAD, CONTENT_PAD, CONTENT_PAD, CONTENT_PAD)
        self._main_layout.setSpacing(0)

        scroll.setWidget(self._content)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    def load_patient(self, patient: dict):
        self._patient = patient

        # Clear previous content
        while self._main_layout.count():
            item = self._main_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Back button
        back_btn = _pill_btn("Back to Patients", filled=False, color=BLUE)
        back_btn.clicked.connect(self.back_requested.emit)
        self._main_layout.addWidget(back_btn, alignment=Qt.AlignLeft)
        self._main_layout.addSpacing(20)

        # Header row
        header_row = QHBoxLayout()
        name = patient.get("name", "Unknown")
        pid  = patient.get("patient_id", "")

        initials = "".join(p[0].upper() for p in name.split()[:2])
        avatar = QLabel(initials)
        avatar.setFixedSize(56, 56)
        avatar.setAlignment(Qt.AlignCenter)
        avatar.setStyleSheet(
            f"background: {BLUE_TINT}; color: {BLUE}; border-radius: 28px; "
            f"font-size: 18px; font-weight: 700; font-family: '{FONT_FAMILY}';"
        )

        info_block = QVBoxLayout()
        info_block.setSpacing(2)
        info_block.setContentsMargins(16, 0, 0, 0)
        info_block.addWidget(_label(name, 20, TEXT, bold=True))
        info_block.addWidget(_label(f"ID: {pid}  |  {patient.get('gender', '')}  |  Age {patient.get('age', '')}", 11, TEXT_SEC))

        header_row.addWidget(avatar)
        header_row.addLayout(info_block)
        header_row.addStretch()

        scan_btn = _pill_btn("New Scan", height=40)
        scan_btn.clicked.connect(lambda: self.new_scan_requested.emit(self._patient))
        header_row.addWidget(scan_btn, alignment=Qt.AlignBottom)

        self._main_layout.addLayout(header_row)
        self._main_layout.addSpacing(24)

        # Info cards row
        info_row = QHBoxLayout()
        info_row.setSpacing(16)

        def info_card(title, value):
            card = QFrame()
            card.setStyleSheet(
                f"background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 12px;"
            )
            cl = QVBoxLayout(card)
            cl.setContentsMargins(16, 14, 16, 14)
            cl.setSpacing(4)
            cl.addWidget(_label(title.upper(), 10, TEXT_SEC))
            cl.addWidget(_label(value or "N/A", 13, TEXT, bold=True))
            return card

        info_row.addWidget(info_card("Contact",    patient.get("contact", "")))
        info_row.addWidget(info_card("Total Scans", str(patient.get("total_scans", 0))))
        info_row.addWidget(info_card("Last Scan",   _fmt_date(patient.get("last_scan_date"))))
        info_row.addWidget(info_card("Last Result", patient.get("last_diagnosis", "None")))
        self._main_layout.addLayout(info_row)
        self._main_layout.addSpacing(24)

        # Symptoms
        if patient.get("symptoms"):
            self._main_layout.addWidget(_label("Current Symptoms", 13, TEXT, bold=True))
            self._main_layout.addSpacing(8)
            symp = QLabel(patient["symptoms"])
            symp.setWordWrap(True)
            symp.setStyleSheet(
                f"background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 12px; "
                f"padding: 14px 16px; font-size: {FONT_BODY}px; color: {TEXT}; font-family: '{FONT_FAMILY}';"
            )
            self._main_layout.addWidget(symp)
            self._main_layout.addSpacing(20)

        # Medical history
        if patient.get("medical_history"):
            self._main_layout.addWidget(_label("Medical History", 13, TEXT, bold=True))
            self._main_layout.addSpacing(8)
            hist = QLabel(patient["medical_history"])
            hist.setWordWrap(True)
            hist.setStyleSheet(
                f"background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 12px; "
                f"padding: 14px 16px; font-size: {FONT_BODY}px; color: {TEXT}; font-family: '{FONT_FAMILY}';"
            )
            self._main_layout.addWidget(hist)
            self._main_layout.addSpacing(20)

        # Scan history
        self._main_layout.addWidget(_label("Scan History", 13, TEXT, bold=True))
        self._main_layout.addSpacing(12)

        history_frame = QFrame()
        history_frame.setStyleSheet(
            f"background: {SURFACE}; border: 1px solid {BORDER}; border-radius: {CARD_RADIUS}px;"
        )
        history_layout = QVBoxLayout(history_frame)
        history_layout.setContentsMargins(0, 0, 0, 0)
        history_layout.setSpacing(0)

        try:
            scans = db.get_scans_for_patient(pid)
        except Exception:
            scans = []

        if not scans:
            empty = _label("No scans recorded for this patient.", 12, TEXT_SEC)
            empty.setAlignment(Qt.AlignCenter)
            empty.setContentsMargins(0, 32, 0, 32)
            history_layout.addWidget(empty)
        else:
            for i, scan in enumerate(scans):
                row = ScanHistoryRow(scan)
                history_layout.addWidget(row)
                if i < len(scans) - 1:
                    history_layout.addWidget(_divider())

        self._main_layout.addWidget(history_frame)
        self._main_layout.addStretch()


# ---------------------------------------------------------------------------
# Patients List Panel
# ---------------------------------------------------------------------------

class PatientsListPanel(QWidget):
    patient_selected = Signal(dict)

    def __init__(self, doctor: dict, parent=None):
        super().__init__(parent)
        self.doctor = doctor
        self.setStyleSheet(f"background: {BG};")
        self._all_patients = []
        self._build_ui()
        self.refresh()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(CONTENT_PAD, CONTENT_PAD, CONTENT_PAD, CONTENT_PAD)
        layout.setSpacing(0)

        header_row = QHBoxLayout()
        header_block = QVBoxLayout()
        header_block.addWidget(_label("Patient Records", 22, TEXT, bold=True))
        header_block.addSpacing(4)
        header_block.addWidget(_label("Search and view all patient records and scan history.", 12, TEXT_SEC))

        self.btn_refresh = _pill_btn("Refresh", filled=False)
        self.btn_refresh.clicked.connect(self.refresh)

        header_row.addLayout(header_block)
        header_row.addStretch()
        header_row.addWidget(self.btn_refresh, alignment=Qt.AlignBottom)
        layout.addLayout(header_row)
        layout.addSpacing(24)

        self.search = QLineEdit()
        self.search.setPlaceholderText("Search by patient ID or name...")
        self.search.setFixedHeight(40)
        self.search.setStyleSheet(f"""
            QLineEdit {{
                background: {SURFACE}; border: 1.5px solid {BORDER};
                border-radius: 20px; padding: 0 16px;
                font-size: {FONT_BODY}px; color: {TEXT}; font-family: '{FONT_FAMILY}';
            }}
            QLineEdit:focus {{ border-color: {BLUE}; }}
        """)
        self.search.textChanged.connect(self._on_search)
        layout.addWidget(self.search)
        layout.addSpacing(16)

        self.table = QTableWidget()
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels([
            "Patient ID", "Name", "Age", "Gender",
            "Last Scan", "Last Diagnosis", "Total Scans"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.setShowGrid(False)
        self.table.setCursor(QCursor(Qt.PointingHandCursor))
        self.table.setStyleSheet(f"""
            QTableWidget {{
                background: {SURFACE}; border: 1px solid {BORDER};
                border-radius: {CARD_RADIUS}px; outline: none;
            }}
            QTableWidget::item {{
                padding: 12px 16px; border-bottom: 1px solid {BORDER};
                font-size: {FONT_BODY}px; color: {TEXT}; font-family: '{FONT_FAMILY}';
            }}
            QTableWidget::item:selected {{ background: {BLUE_TINT}; color: {BLUE}; }}
            QHeaderView::section {{
                background: {BG}; color: {TEXT_SEC};
                font-size: {FONT_LABEL}px; font-weight: 600;
                padding: 12px 16px; border: none;
                border-bottom: 1px solid {BORDER}; font-family: '{FONT_FAMILY}';
            }}
        """)
        self.table.cellDoubleClicked.connect(self._on_row_double_clicked)
        layout.addWidget(self.table, 1)

    def refresh(self):
        try:
            self._all_patients = db.get_all_patients(self.doctor["doctor_id"])
        except Exception as e:
            print(f"Patients load error: {e}")
            self._all_patients = []
        self._populate(self._all_patients)

    def _on_search(self, text):
        text = text.strip()
        if not text:
            self._populate(self._all_patients)
            return
        try:
            results = db.search_patients(text, self.doctor["doctor_id"])
        except Exception:
            results = [
                p for p in self._all_patients
                if text.lower() in p.get("name", "").lower()
                or text.lower() in p.get("patient_id", "").lower()
            ]
        self._populate(results)

    def _populate(self, patients: list):
        self.table.setRowCount(0)
        for p in patients:
            r = self.table.rowCount()
            self.table.insertRow(r)

            diagnosis = p.get("last_diagnosis", "")
            values = [
                p.get("patient_id", ""),
                p.get("name", ""),
                str(p.get("age", "")),
                p.get("gender", ""),
                _fmt_date(p.get("last_scan_date")),
                diagnosis or "No scans",
                str(p.get("total_scans", 0)),
            ]
            for col, val in enumerate(values):
                item = QTableWidgetItem(val)
                item.setTextAlignment(Qt.AlignVCenter | Qt.AlignLeft)
                item.setData(Qt.UserRole, p)
                if col == 5 and diagnosis:
                    item.setForeground(
                        QColor(RED if diagnosis.upper() == "PNEUMONIA" else GREEN)
                    )
                self.table.setItem(r, col, item)

    def _on_row_double_clicked(self, row, col):
        item = self.table.item(row, 0)
        if item:
            patient = item.data(Qt.UserRole)
            if patient:
                self.patient_selected.emit(patient)


# ---------------------------------------------------------------------------
# Patients Panel (container with list + profile views)
# ---------------------------------------------------------------------------

class PatientsPanel(QWidget):
    new_scan_requested = Signal(dict)   # emits patient dict

    def __init__(self, doctor: dict, parent=None):
        super().__init__(parent)
        self.doctor = doctor
        self.setStyleSheet(f"background: {BG};")

        self.stack = QStackedWidget()

        self.list_panel    = PatientsListPanel(doctor)
        self.profile_panel = PatientProfilePanel()

        self.stack.addWidget(self.list_panel)    # index 0
        self.stack.addWidget(self.profile_panel) # index 1

        self.list_panel.patient_selected.connect(self._open_profile)
        self.profile_panel.back_requested.connect(self._back_to_list)
        self.profile_panel.new_scan_requested.connect(self.new_scan_requested.emit)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self.stack)

    def open_patient(self, patient: dict):
        """Called externally (e.g. from dashboard activity click)."""
        self._open_profile(patient)

    def refresh(self):
        self.list_panel.refresh()

    def _open_profile(self, patient: dict):
        self.profile_panel.load_patient(patient)
        self.stack.setCurrentIndex(1)

    def _back_to_list(self):
        self.list_panel.refresh()
        self.stack.setCurrentIndex(0)