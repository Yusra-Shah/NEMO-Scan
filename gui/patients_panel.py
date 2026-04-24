"""
NEMO Scan - Patient Records Panel
Location: gui/patients_panel.py

Loads real patient data from MongoDB.
Supports search, patient profile view, inline editing, and scan management.
"""

from datetime import datetime, timezone

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QTableWidget, QTableWidgetItem, QHeaderView, QLineEdit,
    QPushButton, QScrollArea, QSizePolicy, QStackedWidget,
    QTextEdit, QMessageBox,
)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QColor, QCursor, QFont

import database.db as db

from gui.styles import (
    SURFACE, BORDER, TEXT, TEXT_SEC, BLUE, RED, GREEN, YELLOW,
    BLUE_TINT, RED_TINT, GREEN_TINT, BG, FONT_FAMILY, CONTENT_PAD,
    FONT_BODY, FONT_LABEL, FONT_HEADER, CARD_RADIUS, BTN_RADIUS
)


# ---------------------------------------------------------------------------
# Module-level helpers
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
        hover_bg = BLUE_TINT if color == BLUE else RED_TINT if color == RED else "#F1F3F4"
        btn.setStyleSheet(f"""
            QPushButton {{
                background: {SURFACE}; color: {color};
                border: 1.5px solid {color};
                border-radius: {height // 2}px; padding: 0 18px; font-weight: 600;
            }}
            QPushButton:hover {{ background: {hover_bg}; }}
        """)
    return btn


def _input_style() -> str:
    return (
        f"QLineEdit {{ background: {SURFACE}; border: 1.5px solid {BORDER}; "
        f"border-radius: 8px; padding: 0 12px; font-size: {FONT_BODY}px; "
        f"color: {TEXT}; font-family: '{FONT_FAMILY}'; }}"
        f"QLineEdit:focus {{ border-color: {BLUE}; }}"
    )


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
# Scan History Row
# ---------------------------------------------------------------------------

class ScanHistoryRow(QFrame):
    view_requested   = Signal(dict)
    delete_requested = Signal(str)   # emits scan_id

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
        scan_id    = scan.get("scan_id", "")

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
        layout.setContentsMargins(16, 0, 12, 0)
        layout.setSpacing(12)

        info = QVBoxLayout()
        info.setSpacing(2)
        date_lbl   = _label(_fmt_date(scan_date), FONT_BODY, TEXT, bold=True)
        detail     = f"{severity}  |  {subtype}" if severity and subtype else severity or subtype or ""
        detail_lbl = _label(detail, 10, TEXT_SEC)
        info.addWidget(date_lbl)
        info.addWidget(detail_lbl)
        layout.addLayout(info)
        layout.addStretch()

        conf_lbl = _label(f"{confidence * 100:.1f}%", FONT_LABEL, TEXT_SEC)
        layout.addWidget(conf_lbl)
        layout.addSpacing(8)

        badge = QLabel(prediction.capitalize())
        badge.setFixedHeight(22)
        badge.setContentsMargins(10, 0, 10, 0)
        badge.setAlignment(Qt.AlignCenter)
        badge.setStyleSheet(
            f"background: {badge_bg}; color: {badge_fg}; border-radius: 11px; "
            f"font-size: {FONT_LABEL}px; font-weight: 600; font-family: '{FONT_FAMILY}';"
        )
        layout.addWidget(badge)
        layout.addSpacing(8)

        del_btn = QPushButton("✕")
        del_btn.setFixedSize(28, 28)
        del_btn.setCursor(QCursor(Qt.PointingHandCursor))
        del_btn.setToolTip("Remove this scan")
        del_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent; color: {TEXT_SEC};
                border: none; border-radius: 14px;
                font-size: 11px; font-weight: 700;
            }}
            QPushButton:hover {{ background: {RED_TINT}; color: {RED}; }}
        """)
        del_btn.clicked.connect(lambda: self.delete_requested.emit(scan_id))
        layout.addWidget(del_btn)

    def mousePressEvent(self, event):
        self.view_requested.emit(self._scan)
        super().mousePressEvent(event)


# ---------------------------------------------------------------------------
# Patient Profile Panel
# ---------------------------------------------------------------------------

class PatientProfilePanel(QWidget):
    back_requested     = Signal()
    new_scan_requested = Signal(dict)

    def __init__(self, doctor: dict = None, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background: {BG};")
        self._patient = None
        self._doctor  = doctor or {}
        self._toast   = None
        self._build_ui()

    # ------------------------------------------------------------------
    # Scaffold (called once)
    # ------------------------------------------------------------------

    def _build_ui(self):
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.NoFrame)
        self._scroll.setStyleSheet("background: transparent; border: none;")

        self._content = QWidget()
        self._content.setStyleSheet(f"background: {BG};")
        self._main_layout = QVBoxLayout(self._content)
        self._main_layout.setContentsMargins(CONTENT_PAD, CONTENT_PAD, CONTENT_PAD, CONTENT_PAD)
        self._main_layout.setSpacing(0)

        self._scroll.setWidget(self._content)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self._scroll)

    # ------------------------------------------------------------------
    # Public entry-point — rebuilds the whole profile page
    # ------------------------------------------------------------------

    def load_patient(self, patient: dict, doctor: dict = None):
        self._patient = patient
        if doctor is not None:
            self._doctor = doctor

        # Replace content widget (QScrollArea owns + deletes the old one)
        self._content = QWidget()
        self._content.setStyleSheet(f"background: {BG};")
        self._main_layout = QVBoxLayout(self._content)
        self._main_layout.setContentsMargins(CONTENT_PAD, CONTENT_PAD, CONTENT_PAD, CONTENT_PAD)
        self._main_layout.setSpacing(0)
        self._scroll.setWidget(self._content)

        pid  = patient.get("patient_id", "")
        name = patient.get("name", "Unknown")

        # ── Toast ──────────────────────────────────────────────────────
        self._toast = QLabel("")
        self._toast.setAlignment(Qt.AlignCenter)
        self._toast.setFixedHeight(36)
        self._toast.setVisible(False)
        self._toast.setStyleSheet(
            f"background: {GREEN_TINT}; color: {GREEN}; border-radius: 8px; "
            f"font-size: {FONT_BODY}px; font-weight: 600; font-family: '{FONT_FAMILY}';"
        )
        self._main_layout.addWidget(self._toast)
        self._main_layout.addSpacing(4)

        # ── Back button ────────────────────────────────────────────────
        back_btn = _pill_btn("Back to Patients", filled=False, color=BLUE)
        back_btn.clicked.connect(self.back_requested.emit)
        self._main_layout.addWidget(back_btn, alignment=Qt.AlignLeft)
        self._main_layout.addSpacing(20)

        # ── Header row ─────────────────────────────────────────────────
        header_row = QHBoxLayout()
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
        info_block.addWidget(_label(
            f"ID: {pid}  |  {patient.get('gender', '')}  |  Age {patient.get('age', '')}",
            11, TEXT_SEC,
        ))

        header_row.addWidget(avatar)
        header_row.addLayout(info_block)
        header_row.addStretch()

        edit_info_btn  = _pill_btn("Edit Info", filled=False, color=BLUE, height=40)
        scan_btn       = _pill_btn("New Scan", height=40)
        scan_btn.clicked.connect(lambda: self.new_scan_requested.emit(self._patient))

        deactivate_btn = QPushButton("Deactivate Patient")
        deactivate_btn.setFixedHeight(40)
        deactivate_btn.setCursor(QCursor(Qt.PointingHandCursor))
        deactivate_btn.setFont(QFont(FONT_FAMILY, 11))
        deactivate_btn.setStyleSheet(f"""
            QPushButton {{
                background: {SURFACE}; color: {RED};
                border: 1.5px solid {RED};
                border-radius: 20px; padding: 0 18px; font-weight: 600;
            }}
            QPushButton:hover {{ background: {RED_TINT}; }}
        """)
        deactivate_btn.clicked.connect(lambda: self._confirm_deactivate(pid))

        header_row.addWidget(edit_info_btn, alignment=Qt.AlignBottom)
        header_row.addSpacing(8)
        header_row.addWidget(scan_btn, alignment=Qt.AlignBottom)
        header_row.addSpacing(8)
        header_row.addWidget(deactivate_btn, alignment=Qt.AlignBottom)
        self._main_layout.addLayout(header_row)
        self._main_layout.addSpacing(12)

        # ── Edit Info form (collapsed by default) ──────────────────────
        self._info_edit_frame = self._make_info_edit_form(patient)
        self._info_edit_frame.setVisible(False)
        self._main_layout.addWidget(self._info_edit_frame)
        self._main_layout.addSpacing(12)
        edit_info_btn.clicked.connect(
            lambda: self._info_edit_frame.setVisible(
                not self._info_edit_frame.isVisible()
            )
        )

        # ── Info cards (read-only computed stats) ──────────────────────
        info_row = QHBoxLayout()
        info_row.setSpacing(16)

        def _info_card(title, value):
            card = QFrame()
            card.setStyleSheet(
                f"QFrame {{ background: {SURFACE}; border: none; border-radius: 12px; }}"
                f"QLabel {{ background-color: transparent; border: none; }}"
            )
            cl = QVBoxLayout(card)
            cl.setContentsMargins(16, 14, 16, 14)
            cl.setSpacing(4)
            cl.addWidget(_label(title.upper(), 10, TEXT_SEC))
            cl.addWidget(_label(value or "N/A", 13, TEXT, bold=True))
            return card

        info_row.addWidget(_info_card("Contact",     patient.get("contact", "")))
        info_row.addWidget(_info_card("Total Scans", str(patient.get("total_scans", 0))))
        info_row.addWidget(_info_card("Last Scan",   _fmt_date(patient.get("last_scan_date"))))
        info_row.addWidget(_info_card("Last Result", patient.get("last_diagnosis") or "None"))
        self._main_layout.addLayout(info_row)
        self._main_layout.addSpacing(24)

        # ── Editable text sections ─────────────────────────────────────
        self._main_layout.addWidget(
            self._make_editable_text_section(
                "Current Symptoms", patient.get("symptoms", ""), "symptoms"
            )
        )
        self._main_layout.addSpacing(16)

        self._main_layout.addWidget(
            self._make_editable_text_section(
                "Medical History", patient.get("medical_history", ""), "medical_history"
            )
        )
        self._main_layout.addSpacing(24)

        # ── Scan history ───────────────────────────────────────────────
        self._main_layout.addWidget(_label("Scan History", 13, TEXT, bold=True))
        self._main_layout.addSpacing(12)

        history_frame = QFrame()
        history_frame.setStyleSheet(
            f"QFrame {{ background: {SURFACE}; border: 1px solid {BORDER}; "
            f"border-radius: {CARD_RADIUS}px; }}"
            f"QLabel {{ background-color: transparent; border: none; }}"
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
                row.delete_requested.connect(self._confirm_delete_scan)
                history_layout.addWidget(row)
                if i < len(scans) - 1:
                    history_layout.addWidget(_divider())

        self._main_layout.addWidget(history_frame)
        self._main_layout.addStretch()

    # ------------------------------------------------------------------
    # Edit Info form builder
    # ------------------------------------------------------------------

    def _make_info_edit_form(self, patient: dict) -> QFrame:
        frame = QFrame()
        frame.setStyleSheet(
            f"QFrame {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 12px; }}"
            f"QLabel {{ background-color: transparent; border: none; }}"
        )
        fl = QVBoxLayout(frame)
        fl.setContentsMargins(20, 16, 20, 16)
        fl.setSpacing(10)

        fl.addWidget(_label("Edit Patient Info", 13, TEXT, bold=True))

        # Name (full width)
        fl.addWidget(_label("Name", 10, TEXT_SEC))
        self._edit_name = QLineEdit(patient.get("name", ""))
        self._edit_name.setFixedHeight(36)
        self._edit_name.setStyleSheet(_input_style())
        fl.addWidget(self._edit_name)

        # Contact | Age | Gender
        row2 = QHBoxLayout()
        row2.setSpacing(12)

        col_c = QVBoxLayout()
        col_c.setSpacing(4)
        col_c.addWidget(_label("Contact", 10, TEXT_SEC))
        self._edit_contact = QLineEdit(patient.get("contact", ""))
        self._edit_contact.setFixedHeight(36)
        self._edit_contact.setStyleSheet(_input_style())
        col_c.addWidget(self._edit_contact)

        col_a = QVBoxLayout()
        col_a.setSpacing(4)
        col_a.addWidget(_label("Age", 10, TEXT_SEC))
        self._edit_age = QLineEdit(str(patient.get("age", "")))
        self._edit_age.setFixedHeight(36)
        self._edit_age.setStyleSheet(_input_style())
        col_a.addWidget(self._edit_age)

        col_g = QVBoxLayout()
        col_g.setSpacing(4)
        col_g.addWidget(_label("Gender", 10, TEXT_SEC))
        self._edit_gender = QLineEdit(patient.get("gender", ""))
        self._edit_gender.setFixedHeight(36)
        self._edit_gender.setStyleSheet(_input_style())
        col_g.addWidget(self._edit_gender)

        row2.addLayout(col_c, 2)
        row2.addLayout(col_a, 1)
        row2.addLayout(col_g, 1)
        fl.addLayout(row2)

        btn_row = QHBoxLayout()
        cancel_btn = _pill_btn("Cancel",       filled=False, color=TEXT_SEC, height=34)
        save_btn   = _pill_btn("Save Changes", filled=True,  color=BLUE,    height=34)
        btn_row.addStretch()
        btn_row.addWidget(cancel_btn)
        btn_row.addSpacing(8)
        btn_row.addWidget(save_btn)
        fl.addLayout(btn_row)

        cancel_btn.clicked.connect(lambda: frame.setVisible(False))
        save_btn.clicked.connect(self._save_info)

        return frame

    # ------------------------------------------------------------------
    # Editable text section builder (symptoms / medical history)
    # ------------------------------------------------------------------

    def _make_editable_text_section(self, title: str, text: str, field_key: str) -> QWidget:
        section = QWidget()
        section.setStyleSheet("background: transparent;")
        vl = QVBoxLayout(section)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.setSpacing(8)

        # Title row + toggle buttons
        title_row = QHBoxLayout()
        title_row.addWidget(_label(title, 13, TEXT, bold=True))
        title_row.addStretch()

        edit_btn   = _pill_btn("Edit",   filled=False, color=BLUE,    height=30)
        cancel_btn = _pill_btn("Cancel", filled=False, color=TEXT_SEC, height=30)
        save_btn   = _pill_btn("Save",   filled=True,  color=BLUE,    height=30)
        cancel_btn.setVisible(False)
        save_btn.setVisible(False)
        title_row.addWidget(edit_btn)
        title_row.addWidget(cancel_btn)
        title_row.addWidget(save_btn)
        vl.addLayout(title_row)

        # Read-only label
        read_lbl = QLabel(text if text else "(empty)")
        read_lbl.setWordWrap(True)
        read_lbl.setStyleSheet(
            f"background: {SURFACE}; border-radius: 12px; padding: 14px 16px; "
            f"font-size: {FONT_BODY}px; color: {TEXT if text else TEXT_SEC}; "
            f"font-family: '{FONT_FAMILY}';"
        )
        vl.addWidget(read_lbl)

        # Edit area (hidden until Edit is clicked)
        edit_area = QTextEdit()
        edit_area.setPlainText(text or "")
        edit_area.setFixedHeight(100)
        edit_area.setVisible(False)
        edit_area.setStyleSheet(
            f"QTextEdit {{ background: {SURFACE}; border: 1.5px solid {BORDER}; "
            f"border-radius: 12px; padding: 10px 14px; "
            f"font-size: {FONT_BODY}px; color: {TEXT}; font-family: '{FONT_FAMILY}'; }}"
            f"QTextEdit:focus {{ border-color: {BLUE}; }}"
        )
        vl.addWidget(edit_area)

        def on_edit():
            read_lbl.setVisible(False)
            edit_area.setVisible(True)
            edit_btn.setVisible(False)
            cancel_btn.setVisible(True)
            save_btn.setVisible(True)

        def on_cancel():
            edit_area.setPlainText(text or "")
            edit_area.setVisible(False)
            read_lbl.setVisible(True)
            edit_btn.setVisible(True)
            cancel_btn.setVisible(False)
            save_btn.setVisible(False)

        def on_save():
            new_text  = edit_area.toPlainText().strip()
            pid       = self._patient.get("patient_id", "")
            doc_id    = self._doctor.get("doctor_id", "")
            doc_name  = self._doctor.get("name", "")
            try:
                db.update_patient_info(pid, {field_key: new_text}, doc_id, doc_name)
                self._reload()
                self._show_toast(f"{title} updated.")
            except Exception as e:
                self._show_error(f"Save failed: {e}")

        edit_btn.clicked.connect(on_edit)
        cancel_btn.clicked.connect(on_cancel)
        save_btn.clicked.connect(on_save)

        return section

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _save_info(self):
        name    = self._edit_name.text().strip()
        contact = self._edit_contact.text().strip()
        age_str = self._edit_age.text().strip()
        gender  = self._edit_gender.text().strip()

        if not name:
            self._show_error("Name cannot be empty.")
            return
        if contact and not contact.isdigit():
            self._show_error("Contact must be numeric (digits only).")
            return
        if not age_str.isdigit() or int(age_str) <= 0:
            self._show_error("Age must be a positive integer.")
            return

        reply = QMessageBox.question(
            self, "Confirm Save",
            f"Save changes to patient '{name}'?",
            QMessageBox.Yes | QMessageBox.Cancel,
            QMessageBox.Yes,
        )
        if reply != QMessageBox.Yes:
            return

        pid      = self._patient.get("patient_id", "")
        doc_id   = self._doctor.get("doctor_id", "")
        doc_name = self._doctor.get("name", "")
        try:
            db.update_patient_info(
                pid,
                {"name": name, "contact": contact, "age": int(age_str), "gender": gender},
                doc_id,
                doc_name,
            )
            self._reload()
            self._show_toast("Patient info updated successfully.")
        except Exception as e:
            self._show_error(f"Save failed: {e}")

    def _confirm_delete_scan(self, scan_id: str):
        reply = QMessageBox.warning(
            self,
            "Remove Scan",
            "Are you sure you want to remove this scan?\nThis cannot be undone.",
            QMessageBox.Yes | QMessageBox.Cancel,
            QMessageBox.Cancel,
        )
        if reply != QMessageBox.Yes:
            return
        doc_id   = self._doctor.get("doctor_id", "")
        doc_name = self._doctor.get("name", "")
        try:
            db.deactivate_scan(scan_id, doc_id, doc_name)
            self._reload()
            self._show_toast("Scan removed.")
        except Exception as e:
            self._show_error(f"Could not remove scan: {e}")

    def _confirm_deactivate(self, patient_id: str):
        reply = QMessageBox.warning(
            self,
            "Deactivate Patient",
            "Are you sure you want to deactivate this patient?\nThis cannot be undone from the app.",
            QMessageBox.Yes | QMessageBox.Cancel,
            QMessageBox.Cancel,
        )
        if reply == QMessageBox.Yes:
            doc_id = self._doctor.get("doctor_id", "")
            try:
                db.deactivate_patient(patient_id, doc_id)
            except Exception as e:
                print(f"Deactivate error: {e}")
            self.back_requested.emit()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reload(self):
        """Re-fetch patient from DB and rebuild the profile page in-place."""
        try:
            fresh = db.get_patient_by_id(self._patient["patient_id"])
            if fresh:
                self._patient = fresh
        except Exception:
            pass
        self.load_patient(self._patient, self._doctor)

    def _show_toast(self, message: str, success: bool = True):
        if not self._toast:
            return
        color = GREEN if success else RED
        tint  = GREEN_TINT if success else RED_TINT
        self._toast.setText(message)
        self._toast.setStyleSheet(
            f"background: {tint}; color: {color}; border-radius: 8px; "
            f"font-size: {FONT_BODY}px; font-weight: 600; font-family: '{FONT_FAMILY}';"
        )
        self._toast.setVisible(True)
        QTimer.singleShot(3000, self._auto_hide_toast)

    def _auto_hide_toast(self):
        try:
            if self._toast:
                self._toast.setVisible(False)
        except RuntimeError:
            pass

    def _show_error(self, message: str):
        QMessageBox.critical(self, "Error", message)


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
# Patients Panel (container: list ↔ profile)
# ---------------------------------------------------------------------------

class PatientsPanel(QWidget):
    new_scan_requested = Signal(dict)

    def __init__(self, doctor: dict, parent=None):
        super().__init__(parent)
        self.doctor = doctor
        self.setStyleSheet(f"background: {BG};")

        self.stack = QStackedWidget()

        self.list_panel    = PatientsListPanel(doctor)
        self.profile_panel = PatientProfilePanel(doctor=doctor)

        self.stack.addWidget(self.list_panel)     # index 0
        self.stack.addWidget(self.profile_panel)  # index 1

        self.list_panel.patient_selected.connect(self._open_profile)
        self.profile_panel.back_requested.connect(self._back_to_list)
        self.profile_panel.new_scan_requested.connect(self.new_scan_requested.emit)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self.stack)

    def showEvent(self, event):
        super().showEvent(event)
        self.list_panel.refresh()

    def open_patient(self, patient: dict):
        """Called externally (e.g. from dashboard activity click)."""
        self._open_profile(patient)

    def refresh(self):
        self.list_panel.refresh()

    def _open_profile(self, patient: dict):
        self.profile_panel.load_patient(patient, self.doctor)
        self.stack.setCurrentIndex(1)

    def _back_to_list(self):
        self.list_panel.refresh()
        self.stack.setCurrentIndex(0)
