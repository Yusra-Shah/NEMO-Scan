"""
NEMO Scan - Main Window
Location: gui/main_window.py

Main application window after login.
Contains the fixed sidebar and the content area that swaps between panels.
"""

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QPushButton, QStackedWidget, QFrame,
    QSizePolicy, QSpacerItem
)
from PySide6.QtCore import Qt, QSize, Signal
from PySide6.QtGui import QFont, QCursor, QIcon

from gui.styles import (
    SURFACE, BORDER, TEXT, TEXT_SEC, BLUE, BLUE_TINT,
    BG, SIDEBAR_W, FONT_FAMILY, FONT_BODY, FONT_LABEL,
    FONT_HEADER, MAIN_STYLE
)
from gui.dashboard import DashboardPanel
from gui.patients_panel import PatientsPanel
from gui.models_panel import ModelsPanel
from gui.scan_panel import ScanDialog


class Sidebar(QWidget):
    """Fixed left sidebar with navigation."""

    nav_clicked = Signal(str)

    def __init__(self, doctor: dict, parent=None):
        super().__init__(parent)
        self.setObjectName("sidebar")
        self.setFixedWidth(SIDEBAR_W)
        self.doctor = doctor
        self._active = "dashboard"
        self._nav_btns = {}
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 24, 16, 24)
        layout.setSpacing(4)

        # Logo
        logo_row = QHBoxLayout()
        logo_icon = QLabel("N")
        logo_icon.setFixedSize(32, 32)
        logo_icon.setAlignment(Qt.AlignCenter)
        logo_icon.setStyleSheet(f"""
            background-color: {BLUE};
            color: white;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 800;
            font-family: '{FONT_FAMILY}';
        """)
        logo_text = QLabel("NEMO Scan")
        logo_text.setStyleSheet(f"""
            font-size: 15px;
            font-weight: 700;
            color: {TEXT};
            font-family: '{FONT_FAMILY}';
            margin-left: 8px;
        """)
        logo_row.addWidget(logo_icon)
        logo_row.addWidget(logo_text)
        logo_row.addStretch()
        layout.addLayout(logo_row)
        layout.addSpacing(8)

        version = QLabel("Lung Module  v1.0")
        version.setStyleSheet(
            f"font-size: 10px; color: {TEXT_SEC}; font-family: '{FONT_FAMILY}'; padding-left: 4px;"
        )
        layout.addWidget(version)
        layout.addSpacing(24)

        div = QFrame()
        div.setFrameShape(QFrame.HLine)
        div.setStyleSheet(f"color: {BORDER};")
        layout.addWidget(div)
        layout.addSpacing(16)

        nav_items = [
            ("dashboard", "Dashboard",      "Overview and recent activity"),
            ("scan",      "New Scan",        "Upload and analyze X-ray"),
            ("patients",  "Patient Records", "Patient history and scans"),
            ("models",    "AI Models",       "Model status and accuracy"),
        ]

        for key, label, tooltip in nav_items:
            btn = self._make_nav_btn(key, label, tooltip)
            layout.addWidget(btn)
            self._nav_btns[key] = btn

        layout.addStretch()

        div2 = QFrame()
        div2.setFrameShape(QFrame.HLine)
        div2.setStyleSheet(f"color: {BORDER};")
        layout.addWidget(div2)
        layout.addSpacing(16)

        # Doctor avatar + name + specialization
        name = self.doctor.get("name", "Doctor")
        spec = self.doctor.get("specialization", "")
        initials = "".join(p[0].upper() for p in name.split()[:2])

        doc_row = QHBoxLayout()
        avatar = QLabel(initials)
        avatar.setFixedSize(32, 32)
        avatar.setAlignment(Qt.AlignCenter)
        avatar.setStyleSheet(f"""
            background-color: {BLUE_TINT};
            color: {BLUE};
            border-radius: 16px;
            font-size: 11px;
            font-weight: 700;
            font-family: '{FONT_FAMILY}';
        """)

        info = QVBoxLayout()
        info.setSpacing(0)
        info.setContentsMargins(8, 0, 0, 0)
        doc_name = QLabel(name)
        doc_name.setStyleSheet(
            f"font-size: 12px; font-weight: 600; color: {TEXT}; font-family: '{FONT_FAMILY}';"
        )
        doc_name.setMaximumWidth(150)
        doc_spec = QLabel(spec)
        doc_spec.setStyleSheet(
            f"font-size: 10px; color: {TEXT_SEC}; font-family: '{FONT_FAMILY}';"
        )
        doc_spec.setMaximumWidth(150)
        info.addWidget(doc_name)
        info.addWidget(doc_spec)

        doc_row.addWidget(avatar)
        doc_row.addLayout(info)
        doc_row.addStretch()
        layout.addLayout(doc_row)

        self._set_active("dashboard")

    def _make_nav_btn(self, key: str, label: str, tooltip: str) -> QPushButton:
        btn = QPushButton(f"  {label}")
        btn.setObjectName("nav_btn")
        btn.setFixedHeight(40)
        btn.setToolTip(tooltip)
        btn.setCursor(QCursor(Qt.PointingHandCursor))
        btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {TEXT_SEC};
                border: none;
                border-radius: 10px;
                padding: 0 12px;
                font-size: {FONT_BODY}px;
                font-family: '{FONT_FAMILY}';
                text-align: left;
            }}
            QPushButton:hover {{
                background-color: {BLUE_TINT};
                color: {BLUE};
            }}
        """)
        btn.clicked.connect(lambda _, k=key: self._on_nav_clicked(k))
        return btn

    def _on_nav_clicked(self, key: str):
        self._set_active(key)
        self.nav_clicked.emit(key)

    def _set_active(self, key: str):
        self._active = key
        for k, btn in self._nav_btns.items():
            if k == key:
                btn.setStyleSheet(f"""
                    QPushButton {{
                        background-color: {BLUE_TINT};
                        color: {BLUE};
                        border: none;
                        border-radius: 10px;
                        padding: 0 12px;
                        font-size: {FONT_BODY}px;
                        font-family: '{FONT_FAMILY}';
                        font-weight: 600;
                        text-align: left;
                    }}
                """)
            else:
                btn.setStyleSheet(f"""
                    QPushButton {{
                        background-color: transparent;
                        color: {TEXT_SEC};
                        border: none;
                        border-radius: 10px;
                        padding: 0 12px;
                        font-size: {FONT_BODY}px;
                        font-family: '{FONT_FAMILY}';
                        text-align: left;
                    }}
                    QPushButton:hover {{
                        background-color: {BLUE_TINT};
                        color: {BLUE};
                    }}
                """)


class MainWindow(QMainWindow):

    def __init__(self, doctor: dict, engine=None, parent=None):
        super().__init__(parent)
        self.doctor = doctor
        self.engine = engine
        self.setWindowTitle("NEMO Scan - Neural Engine for Medical Observation")
        self.setMinimumSize(1200, 750)
        self.resize(1400, 900)
        self.setStyleSheet(MAIN_STYLE)
        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        central.setObjectName("main_window")
        self.setCentralWidget(central)

        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.sidebar = Sidebar(self.doctor)
        self.sidebar.nav_clicked.connect(self._on_nav)
        root.addWidget(self.sidebar)

        self.stack = QStackedWidget()
        self.stack.setObjectName("content_area")

        self.dashboard = DashboardPanel(self.doctor)
        self.patients  = PatientsPanel(self.doctor)
        self.models    = ModelsPanel()

        self.stack.addWidget(self.dashboard)  # index 0
        self.stack.addWidget(self.patients)   # index 1
        self.stack.addWidget(self.models)     # index 2

        root.addWidget(self.stack)

        # Wire cross-panel signals
        self.dashboard.new_scan_requested.connect(self._open_scan_dialog)
        self.dashboard.open_patient_profile.connect(self._open_patient_profile)
        self.patients.new_scan_requested.connect(self._open_scan_dialog)

    def _on_nav(self, key: str):
        if key == "dashboard":
            self.stack.setCurrentIndex(0)
        elif key == "scan":
            self._open_scan_dialog()
            self.sidebar._set_active("dashboard")
        elif key == "patients":
            self.stack.setCurrentIndex(1)
        elif key == "models":
            self.stack.setCurrentIndex(2)

    def _open_scan_dialog(self, patient: dict = None):
        dlg = ScanDialog(self.engine, self.doctor, self, patient=patient)
        dlg.exec()
        self.dashboard.refresh()
        self.patients.refresh()

    def _open_patient_profile(self, patient: dict):
        self.stack.setCurrentIndex(1)
        self.patients.open_patient(patient)
        self.sidebar._set_active("patients")