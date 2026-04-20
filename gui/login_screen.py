"""
NEMO Scan - Login Screen
Location: gui/login_screen.py

Two modes: Login (existing doctor) and Register (new doctor account).
Authenticates against MongoDB doctors collection using bcrypt.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QLineEdit, QFrame, QComboBox,
    QGraphicsDropShadowEffect, QSizePolicy
)
from PySide6.QtCore import Qt, Signal, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import QFont, QColor, QCursor

import database.db as db


# ---------------------------------------------------------------------------
# Color constants (inline to avoid circular import with styles.py)
# ---------------------------------------------------------------------------
BG       = "#F8F9FA"
SURFACE  = "#FFFFFF"
BORDER   = "#E8EAED"
TEXT     = "#202124"
TEXT_SEC = "#5F6368"
BLUE     = "#4285F4"
RED      = "#EA4335"
GREEN    = "#34A853"
YELLOW   = "#FBBC05"
FONT     = "Segoe UI"


def _label(text, size=12, color=TEXT, bold=False):
    l = QLabel(text)
    f = QFont(FONT, size)
    f.setBold(bold)
    l.setFont(f)
    l.setStyleSheet(f"color: {color}; background: transparent;")
    return l


def _input(placeholder, echo_mode=QLineEdit.Normal):
    field = QLineEdit()
    field.setPlaceholderText(placeholder)
    field.setEchoMode(echo_mode)
    field.setFixedHeight(44)
    field.setFont(QFont(FONT, 12))
    field.setCursor(QCursor(Qt.IBeamCursor))
    field.setStyleSheet(f"""
        QLineEdit {{
            background: {BG};
            border: 1.5px solid {BORDER};
            border-radius: 22px;
            padding: 0 18px;
            color: {TEXT};
        }}
        QLineEdit:focus {{
            border: 1.5px solid {BLUE};
            background: {SURFACE};
        }}
    """)
    return field


def _pill_btn(text, filled=True, color=BLUE):
    btn = QPushButton(text)
    btn.setFixedHeight(44)
    btn.setFont(QFont(FONT, 12))
    btn.setCursor(QCursor(Qt.PointingHandCursor))
    if filled:
        btn.setStyleSheet(f"""
            QPushButton {{
                background: {color};
                color: white;
                border: none;
                border-radius: 22px;
                padding: 0 24px;
            }}
            QPushButton:hover {{ background: #3367D6; }}
            QPushButton:pressed {{ background: #2A56C6; }}
        """)
    else:
        btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                color: {color};
                border: 1.5px solid {color};
                border-radius: 22px;
                padding: 0 24px;
            }}
            QPushButton:hover {{ background: rgba(66,133,244,0.08); }}
        """)
    return btn


# ---------------------------------------------------------------------------
# Login Card
# ---------------------------------------------------------------------------

class LoginCard(QFrame):
    login_success = Signal(dict)   # emits full doctor dict
    switch_to_register = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build()

    def _build(self):
        self.setFixedWidth(420)
        self.setStyleSheet(f"""
            QFrame {{
                background: {SURFACE};
                border: 1px solid {BORDER};
                border-radius: 20px;
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 44, 40, 44)
        layout.setSpacing(0)

        # wordmark
        wordmark = QHBoxLayout()
        wordmark.setSpacing(0)
        colors = [BLUE, RED, YELLOW, GREEN]
        for i, ch in enumerate("NEMO"):
            lbl = _label(ch, size=26, color=colors[i], bold=True)
            lbl.setAlignment(Qt.AlignCenter)
            wordmark.addWidget(lbl)
        sp = _label(" ", size=26, bold=True)
        wordmark.addWidget(sp)
        for ch in "Scan":
            lbl = _label(ch, size=26, color=TEXT, bold=True)
            wordmark.addWidget(lbl)
        wordmark.addStretch()
        layout.addLayout(wordmark)

        layout.addSpacing(6)
        layout.addWidget(_label("AI-Powered Medical Diagnostic System", 11, TEXT_SEC))
        layout.addSpacing(32)

        layout.addWidget(_label("Sign In", 18, TEXT, bold=True))
        layout.addSpacing(4)
        layout.addWidget(_label("Enter your credentials to access the system.", 11, TEXT_SEC))
        layout.addSpacing(24)

        # fields
        layout.addWidget(_label("Email Address", 11, TEXT_SEC))
        layout.addSpacing(6)
        self.email_field = _input("doctor@hospital.com")
        layout.addWidget(self.email_field)
        layout.addSpacing(14)

        layout.addWidget(_label("Password", 11, TEXT_SEC))
        layout.addSpacing(6)
        self.password_field = _input("Password", QLineEdit.Password)
        layout.addWidget(self.password_field)
        layout.addSpacing(8)

        # error label
        self.error_label = _label("", 11, RED)
        self.error_label.setWordWrap(True)
        self.error_label.hide()
        layout.addWidget(self.error_label)
        layout.addSpacing(8)

        # login button
        self.login_btn = _pill_btn("Sign In")
        self.login_btn.clicked.connect(self._on_login)
        layout.addWidget(self.login_btn)
        layout.addSpacing(16)

        # switch to register
        bottom = QHBoxLayout()
        bottom.addWidget(_label("New to NEMO Scan?", 11, TEXT_SEC))
        bottom.addSpacing(4)
        reg_btn = QPushButton("Create Account")
        reg_btn.setFlat(True)
        reg_btn.setCursor(QCursor(Qt.PointingHandCursor))
        reg_btn.setFont(QFont(FONT, 11))
        reg_btn.setStyleSheet(f"color: {BLUE}; background: transparent; border: none;")
        reg_btn.clicked.connect(self.switch_to_register)
        bottom.addWidget(reg_btn)
        bottom.addStretch()
        layout.addLayout(bottom)

        # enter key triggers login
        self.password_field.returnPressed.connect(self._on_login)
        self.email_field.returnPressed.connect(self._on_login)

    def _on_login(self):
        email = self.email_field.text().strip()
        password = self.password_field.text()

        self.error_label.hide()

        if not email or not password:
            self.error_label.setText("Please enter your email and password.")
            self.error_label.show()
            return

        self.login_btn.setText("Signing in...")
        self.login_btn.setEnabled(False)

        try:
            doctor = db.login_doctor(email, password)
            if doctor:
                self.login_success.emit(doctor)
            else:
                self.error_label.setText("Incorrect email or password.")
                self.error_label.show()
        except Exception as e:
            self.error_label.setText(f"Connection error: {e}")
            self.error_label.show()
        finally:
            self.login_btn.setText("Sign In")
            self.login_btn.setEnabled(True)


# ---------------------------------------------------------------------------
# Register Card
# ---------------------------------------------------------------------------

class RegisterCard(QFrame):
    register_success = Signal(dict)  # emits full doctor dict after registration + auto-login
    switch_to_login = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build()

    def _build(self):
        self.setFixedWidth(420)
        self.setStyleSheet(f"""
            QFrame {{
                background: {SURFACE};
                border: 1px solid {BORDER};
                border-radius: 20px;
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 44, 40, 44)
        layout.setSpacing(0)

        layout.addWidget(_label("Create Account", 18, TEXT, bold=True))
        layout.addSpacing(4)
        layout.addWidget(_label("Register a new doctor account.", 11, TEXT_SEC))
        layout.addSpacing(24)

        layout.addWidget(_label("Full Name", 11, TEXT_SEC))
        layout.addSpacing(6)
        self.name_field = _input("Dr. Jane Smith")
        layout.addWidget(self.name_field)
        layout.addSpacing(14)

        layout.addWidget(_label("Email Address", 11, TEXT_SEC))
        layout.addSpacing(6)
        self.email_field = _input("doctor@hospital.com")
        layout.addWidget(self.email_field)
        layout.addSpacing(14)

        layout.addWidget(_label("Specialization", 11, TEXT_SEC))
        layout.addSpacing(6)
        self.spec_combo = QComboBox()
        self.spec_combo.addItems([
            "General Physician",
            "Pulmonologist",
            "Radiologist",
            "Internal Medicine",
            "Pediatrician",
            "Other",
        ])
        self.spec_combo.setFixedHeight(44)
        self.spec_combo.setFont(QFont(FONT, 12))
        self.spec_combo.setStyleSheet(f"""
            QComboBox {{
                background: {BG};
                border: 1.5px solid {BORDER};
                border-radius: 22px;
                padding: 0 18px;
                color: {TEXT};
            }}
            QComboBox:focus {{ border: 1.5px solid {BLUE}; }}
            QComboBox::drop-down {{ border: none; width: 32px; }}
        """)
        layout.addWidget(self.spec_combo)
        layout.addSpacing(14)

        layout.addWidget(_label("Password", 11, TEXT_SEC))
        layout.addSpacing(6)
        self.password_field = _input("Minimum 6 characters", QLineEdit.Password)
        layout.addWidget(self.password_field)
        layout.addSpacing(14)

        layout.addWidget(_label("Confirm Password", 11, TEXT_SEC))
        layout.addSpacing(6)
        self.confirm_field = _input("Re-enter password", QLineEdit.Password)
        layout.addWidget(self.confirm_field)
        layout.addSpacing(8)

        self.error_label = _label("", 11, RED)
        self.error_label.setWordWrap(True)
        self.error_label.hide()
        layout.addWidget(self.error_label)
        layout.addSpacing(8)

        self.register_btn = _pill_btn("Create Account", color=GREEN)
        self.register_btn.clicked.connect(self._on_register)
        layout.addWidget(self.register_btn)
        layout.addSpacing(16)

        bottom = QHBoxLayout()
        bottom.addWidget(_label("Already have an account?", 11, TEXT_SEC))
        bottom.addSpacing(4)
        login_btn = QPushButton("Sign In")
        login_btn.setFlat(True)
        login_btn.setCursor(QCursor(Qt.PointingHandCursor))
        login_btn.setFont(QFont(FONT, 11))
        login_btn.setStyleSheet(f"color: {BLUE}; background: transparent; border: none;")
        login_btn.clicked.connect(self.switch_to_login)
        bottom.addWidget(login_btn)
        bottom.addStretch()
        layout.addLayout(bottom)

    def _on_register(self):
        name     = self.name_field.text().strip()
        email    = self.email_field.text().strip()
        spec     = self.spec_combo.currentText()
        password = self.password_field.text()
        confirm  = self.confirm_field.text()

        self.error_label.hide()

        if not name or not email or not password:
            self.error_label.setText("All fields are required.")
            self.error_label.show()
            return
        if len(password) < 6:
            self.error_label.setText("Password must be at least 6 characters.")
            self.error_label.show()
            return
        if password != confirm:
            self.error_label.setText("Passwords do not match.")
            self.error_label.show()
            return

        self.register_btn.setText("Creating account...")
        self.register_btn.setEnabled(False)

        try:
            doctor = db.register_doctor(name, email, password, spec)
            # auto-login: fetch session doctor (register_doctor returns safe dict)
            self.register_success.emit(doctor)
        except ValueError as e:
            self.error_label.setText(str(e))
            self.error_label.show()
        except Exception as e:
            self.error_label.setText(f"Error: {e}")
            self.error_label.show()
        finally:
            self.register_btn.setText("Create Account")
            self.register_btn.setEnabled(True)


# ---------------------------------------------------------------------------
# Login Screen (full window widget)
# ---------------------------------------------------------------------------

class LoginScreen(QWidget):
    """
    Full-screen login widget shown before the main window.
    Switches between LoginCard and RegisterCard.
    Emits login_success(doctor_dict) when auth completes.
    """

    login_success = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background: {BG};")
        self._build()

    def _build(self):
        outer = QVBoxLayout(self)
        outer.setAlignment(Qt.AlignCenter)
        outer.setContentsMargins(0, 0, 0, 0)

        self.login_card    = LoginCard()
        self.register_card = RegisterCard()
        self.register_card.hide()

        self.login_card.login_success.connect(self.login_success)
        self.login_card.switch_to_register.connect(self._show_register)

        self.register_card.register_success.connect(self.login_success)
        self.register_card.switch_to_login.connect(self._show_login)

        outer.addWidget(self.login_card, alignment=Qt.AlignCenter)
        outer.addWidget(self.register_card, alignment=Qt.AlignCenter)

        # footer
        footer = _label(
            "NEMO Scan v1.0  |  Prototype  |  Academic Use Only",
            10, TEXT_SEC
        )
        footer.setAlignment(Qt.AlignCenter)
        outer.addSpacing(24)
        outer.addWidget(footer)

    def _show_register(self):
        self.login_card.hide()
        self.register_card.show()

    def _show_login(self):
        self.register_card.hide()
        self.login_card.show()
