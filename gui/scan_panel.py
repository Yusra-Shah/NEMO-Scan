"""
PneumoScan - Holographic Scan Dialog
Location: gui/scan_panel.py
"""

import os
import math
import random

from PySide6.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFrame, QProgressBar, QFileDialog,
    QStackedWidget, QLineEdit, QApplication, QScrollArea,
    QSizePolicy, QMessageBox
)
from PySide6.QtCore import Qt, Signal, QTimer, QThread, QObject
from PySide6.QtGui import (
    QColor, QPainter, QPen, QBrush, QFont, QPixmap, QCursor
)

from gui.styles import (
    HOLO_BG, HOLO_BLUE, HOLO_BLUE_GLOW, HOLO_BLUE_LIGHT,
    HOLO_TEXT, HOLO_TEXT_SEC, HOLO_GREEN, HOLO_RED,
    FONT_FAMILY, FONT_BODY, FONT_LABEL, BTN_RADIUS
)
import database.db as db
from core.report_generator import generate_report


# ---------------------------------------------------------------------------
# Particle / canvas (unchanged)
# ---------------------------------------------------------------------------

class Particle:
    def __init__(self, w, h):
        self.x = random.uniform(0, w)
        self.y = random.uniform(0, h)
        self.vx = random.uniform(-0.3, 0.3)
        self.vy = random.uniform(-0.6, -0.1)
        self.life = random.uniform(0.5, 1.0)
        self.decay = random.uniform(0.003, 0.006)
        self.size = random.uniform(1.2, 2.8)

    def update(self, w, h):
        self.x += self.vx
        self.y += self.vy
        self.life -= self.decay
        if self.life <= 0 or self.y < 0 or self.x < 0 or self.x > w:
            self.x = random.uniform(0, w)
            self.y = h
            self.life = random.uniform(0.5, 1.0)


class ParticleCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.particles = []
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self._timer = QTimer()
        self._timer.timeout.connect(self._tick)
        self._timer.start(33)

    def showEvent(self, e):
        super().showEvent(e)
        w, h = self.width() or 1000, self.height() or 700
        self.particles = [Particle(w, h) for _ in range(70)]

    def _tick(self):
        w, h = self.width(), self.height()
        for p in self.particles:
            p.update(w, h)
        self.update()

    def paintEvent(self, event):
        if not self.particles:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        pen = QPen()
        for i in range(0, len(self.particles) - 1, 6):
            p1 = self.particles[i]
            p2 = self.particles[min(i+1, len(self.particles)-1)]
            dist = math.sqrt((p1.x-p2.x)**2 + (p1.y-p2.y)**2)
            if dist < 90:
                alpha = int((1 - dist/90) * min(p1.life, p2.life) * 100)
                pen.setColor(QColor(66, 133, 244, alpha))
                pen.setWidth(1)
                painter.setPen(pen)
                painter.drawLine(int(p1.x), int(p1.y), int(p2.x), int(p2.y))
        painter.setPen(Qt.NoPen)
        for p in self.particles:
            a = int(p.life * 180)
            painter.setBrush(QBrush(QColor(137, 180, 250, a)))
            r = p.size
            painter.drawEllipse(int(p.x-r), int(p.y-r), int(r*2), int(r*2))
        painter.end()


class SpinnerWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(56, 56)
        self._angle = 0
        self._running = False
        self._timer = QTimer()
        self._timer.timeout.connect(self._tick)

    def start(self):
        self._running = True
        self._timer.start(18)

    def stop(self):
        self._running = False
        self._timer.stop()
        self.update()

    def _tick(self):
        self._angle = (self._angle + 7) % 360
        self.update()

    def paintEvent(self, event):
        if not self._running:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.translate(28, 28)
        for i in range(10):
            angle = self._angle + i * 36
            alpha = int(255 * (i+1) / 10)
            pen = QPen(QColor(66, 133, 244, alpha))
            pen.setWidth(3)
            pen.setCapStyle(Qt.RoundCap)
            painter.setPen(pen)
            rad = math.radians(angle)
            x1 = int(14 * math.cos(rad)); y1 = int(14 * math.sin(rad))
            x2 = int(22 * math.cos(rad)); y2 = int(22 * math.sin(rad))
            painter.drawLine(x1, y1, x2, y2)
        painter.end()


class GlowFrame(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            QFrame {
                background-color: rgba(10, 25, 50, 0.88);
                border: 1px solid rgba(66, 133, 244, 0.38);
                border-radius: 14px;
            }
        """)


class ModelVoteBar(QWidget):
    def __init__(self, display_name, parent=None):
        super().__init__(parent)
        self.setFixedHeight(34)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 3, 0, 3)
        layout.setSpacing(8)

        self.name = QLabel(display_name)
        self.name.setFixedWidth(130)
        self.name.setStyleSheet(f"font-size: 10px; color: {HOLO_TEXT_SEC}; font-family: '{FONT_FAMILY}'; background-color: transparent; border: none;")

        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(5)
        self._set_bar_color(HOLO_BLUE)

        self.pct = QLabel("--")
        self.pct.setFixedWidth(38)
        self.pct.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.pct.setStyleSheet(f"font-size: 10px; color: {HOLO_BLUE_LIGHT}; font-family: '{FONT_FAMILY}'; background-color: transparent; border: none;")

        self.status = QLabel("Waiting")
        self.status.setFixedWidth(55)
        self.status.setAlignment(Qt.AlignCenter)
        self.status.setStyleSheet(f"font-size: 9px; color: {HOLO_TEXT_SEC}; background-color: rgba(66,133,244,0.08); border-radius: 8px; padding: 2px 4px; font-family: '{FONT_FAMILY}';")

        layout.addWidget(self.name)
        layout.addWidget(self.bar, 1)
        layout.addWidget(self.pct)
        layout.addWidget(self.status)

    def _set_bar_color(self, color):
        self.bar.setStyleSheet(
            f"QProgressBar {{ border: none; border-radius: 2px; background-color: rgba(66,133,244,0.12); }}"
            f"QProgressBar::chunk {{ background-color: {color}; border-radius: 2px; }}"
        )

    def set_value(self, prob):
        pct = int(prob * 100)
        self.bar.setValue(pct)
        self.pct.setText(f"{pct}%")
        if prob >= 0.5:
            self._set_bar_color(HOLO_RED)
            self.pct.setStyleSheet(f"font-size: 10px; color: {HOLO_RED}; font-family: '{FONT_FAMILY}'; background-color: transparent; border: none;")
            self.status.setText("Positive")
            self.status.setStyleSheet(f"font-size: 9px; color: {HOLO_RED}; background-color: rgba(234,67,53,0.1); border-radius: 8px; padding: 2px 4px; font-family: '{FONT_FAMILY}';")
        else:
            self._set_bar_color(HOLO_GREEN)
            self.pct.setStyleSheet(f"font-size: 10px; color: {HOLO_GREEN}; font-family: '{FONT_FAMILY}'; background-color: transparent; border: none;")
            self.status.setText("Normal")
            self.status.setStyleSheet(f"font-size: 9px; color: {HOLO_GREEN}; background-color: rgba(52,168,83,0.1); border-radius: 8px; padding: 2px 4px; font-family: '{FONT_FAMILY}';")

    def reset(self):
        self.bar.setValue(0)
        self.pct.setText("--")
        self.status.setText("Waiting")
        self.status.setStyleSheet(
            f"font-size: 9px; color: {HOLO_TEXT_SEC}; background-color: rgba(66,133,244,0.08);"
            f"border-radius: 8px; padding: 2px 4px; font-family: '{FONT_FAMILY}';"
        )
        self._set_bar_color(HOLO_BLUE)


class AnalyticsChart(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._data = {}

    def set_data(self, votes):
        self._data = votes
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        if not self._data:
            painter.setPen(QColor(137, 180, 250, 70))
            painter.setFont(QFont(FONT_FAMILY, 9))
            painter.drawText(self.rect(), Qt.AlignCenter, "Awaiting analysis")
            painter.end()
            return
        items = list(self._data.items())
        n = len(items)
        pad, bot, top = 10, 28, 8
        slot_w = (w - pad*2) / n
        for i, (model, prob) in enumerate(items):
            x = pad + i * slot_w + 2
            bw = slot_w - 4
            bh = max(3, (h - bot - top) * prob)
            by = h - bot - bh
            color = QColor(234, 67, 53, 200) if prob >= 0.5 else QColor(52, 168, 83, 200)
            painter.setBrush(QBrush(color))
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(int(x), int(by), int(bw), int(bh), 3, 3)
            painter.setPen(QColor(137, 180, 250))
            painter.setFont(QFont(FONT_FAMILY, 7))
            painter.drawText(int(x), int(by)-2, int(bw), 12, Qt.AlignCenter, f"{int(prob*100)}%")
            painter.setPen(QColor(100, 140, 200))
            painter.setFont(QFont(FONT_FAMILY, 6))
            painter.drawText(int(x), h-bot+2, int(bw), 24, Qt.AlignCenter | Qt.TextWordWrap, model[:5].upper())
        painter.end()


# ---------------------------------------------------------------------------
# Inference worker (unchanged logic, added timing)
# ---------------------------------------------------------------------------

class InferenceWorker(QObject):
    quick_done = Signal(dict)
    model_done = Signal(str, float)
    full_done  = Signal(dict)
    error      = Signal(str)

    def __init__(self, engine, image_path, heatmaps_dir):
        super().__init__()
        self.engine = engine
        self.image_path = image_path
        self.heatmaps_dir = heatmaps_dir
        self._start_ms = 0

    def run(self):
        import time
        self._start_ms = int(time.time() * 1000)
        try:
            if self.engine is None:
                import random
                time.sleep(0.6)
                self.quick_done.emit({'prediction': 'PNEUMONIA', 'confidence': 0.87, 'probability': 0.87})
                for m in ['resnet50', 'densenet121', 'efficientnet_b4', 'vit_b16', 'inception_v3']:
                    time.sleep(0.4)
                    self.model_done.emit(m, random.uniform(0.75, 0.95))
                elapsed = int(time.time() * 1000) - self._start_ms
                self.full_done.emit({
                    'prediction': 'PNEUMONIA', 'confidence': 0.874, 'severity': 'Moderate',
                    'subtype': 'Bacterial',
                    'model_votes': {
                        'mobilenetv3': 0.87, 'resnet50': 0.91, 'densenet121': 0.88,
                        'efficientnet_b4': 0.85, 'vit_b16': 0.79, 'inception_v3': 0.83,
                        'attention_cnn': 0.86,
                    },
                    'ensemble_prob': 0.874,
                    'heatmap_path': None,
                    'processing_time_ms': elapsed,
                })
                return

            quick = self.engine.predict_quick(self.image_path)
            self.quick_done.emit(quick)

            def on_model(name, prob):
                self.model_done.emit(name, prob)

            result = self.engine.predict_full(
                self.image_path,
                heatmaps_dir=self.heatmaps_dir,
                progress_callback=on_model,
            )
            elapsed = int(time.time() * 1000) - self._start_ms
            result['processing_time_ms'] = elapsed
            self.full_done.emit(result)

        except Exception as e:
            self.error.emit(str(e))


# ---------------------------------------------------------------------------
# Doctor Notes Dialog (shown before PDF generation)
# ---------------------------------------------------------------------------

class _NotesDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Doctor Notes")
        self.setFixedWidth(460)
        self.setModal(True)
        self.setStyleSheet(f"background-color: {HOLO_BG};")
        self._build()

    def _build(self):
        from PySide6.QtWidgets import QTextEdit, QDialogButtonBox
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(12)

        title = QLabel("Add Doctor Notes  (optional)")
        title.setStyleSheet(
            f"font-size: 13px; font-weight: 700; color: {HOLO_BLUE_LIGHT}; "
            f"font-family: '{FONT_FAMILY}'; background-color: transparent; border: none;"
        )
        layout.addWidget(title)

        hint = QLabel("These notes will appear in the PDF report.  Leave blank to skip.")
        hint.setStyleSheet(
            f"font-size: 10px; color: {HOLO_TEXT_SEC}; font-family: '{FONT_FAMILY}'; background-color: transparent; border: none;"
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self._text = QTextEdit()
        self._text.setPlaceholderText(
            "e.g. Patient shows right lower lobe consolidation consistent with bacterial pneumonia. "
            "Recommend antibiotic therapy and follow-up X-ray in 2 weeks."
        )
        self._text.setFixedHeight(120)
        self._text.setFont(QFont(FONT_FAMILY, 10))
        self._text.setStyleSheet(
            f"QTextEdit {{ background-color: rgba(15,35,70,0.9); "
            f"border: 1px solid rgba(66,133,244,0.4); border-radius: 8px; "
            f"padding: 8px; color: {HOLO_TEXT}; font-family: '{FONT_FAMILY}'; }}"
            f"QTextEdit:focus {{ border-color: {HOLO_BLUE}; }}"
        )
        layout.addWidget(self._text)

        btn_row = QHBoxLayout()
        skip_btn = QPushButton("Skip & Generate")
        skip_btn.setFixedHeight(36)
        skip_btn.setCursor(QCursor(Qt.PointingHandCursor))
        skip_btn.setStyleSheet(
            f"QPushButton {{ background-color: transparent; color: {HOLO_TEXT_SEC}; "
            f"border: 1px solid rgba(66,133,244,0.3); border-radius: {BTN_RADIUS}px; "
            f"font-size: {FONT_BODY}px; font-family: '{FONT_FAMILY}'; }}"
            f"QPushButton:hover {{ color: {HOLO_BLUE_LIGHT}; }}"
        )
        skip_btn.clicked.connect(self.accept)

        gen_btn = QPushButton("Generate Report")
        gen_btn.setFixedHeight(36)
        gen_btn.setCursor(QCursor(Qt.PointingHandCursor))
        gen_btn.setStyleSheet(
            f"QPushButton {{ background-color: {HOLO_BLUE}; color: white; border: none; "
            f"border-radius: {BTN_RADIUS}px; font-size: {FONT_BODY}px; font-weight: 600; "
            f"font-family: '{FONT_FAMILY}'; }}"
            f"QPushButton:hover {{ background-color: {HOLO_BLUE_GLOW}; }}"
        )
        gen_btn.clicked.connect(self.accept)

        btn_row.addWidget(skip_btn)
        btn_row.addStretch()
        btn_row.addWidget(gen_btn)
        layout.addLayout(btn_row)

    def get_notes(self) -> str:
        return self._text.toPlainText().strip()


# ---------------------------------------------------------------------------
# Scan Dialog
# ---------------------------------------------------------------------------

class ScanDialog(QDialog):
    def __init__(self, engine=None, doctor=None, parent=None, patient=None):
        super().__init__(parent)

        # Accept both dict and legacy string for doctor
        if isinstance(doctor, str):
            self.doctor = {"doctor_id": "", "name": doctor}
        else:
            self.doctor = doctor or {}

        self.patient = patient  # pre-selected patient dict, or None
        self.engine = engine
        self.image_path = None
        self.result = None
        self._thread = None
        self._worker = None
        self._model_bars = {}
        self._models_done = 0
        self._total_models = 6
        self._heatmap_path = None
        self._showing_heatmap = False
        self._saved_scan_id = None

        self.setWindowTitle("PneumoScan - AI Diagnostic System")
        self.setMinimumSize(900, 600)
        self.resize(1200, 800)
        self.setStyleSheet(f"background-color: {HOLO_BG};")
        self.setModal(True)
        self._build_ui()

        # If patient was passed in, pre-fill and lock the patient ID field
        if self.patient:
            self.patient_id_field.setText(self.patient.get("patient_id", ""))
            self.patient_id_field.setReadOnly(True)
            self.patient_name_lbl.setText(self.patient.get("name", ""))

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        self.canvas = ParticleCanvas(self)
        self.canvas.setGeometry(0, 0, 1200, 800)
        self.canvas.lower()

        wrapper = QWidget()
        wrapper.setStyleSheet("background-color: transparent;")
        wl = QVBoxLayout(wrapper)
        wl.setContentsMargins(28, 16, 28, 16)
        wl.setSpacing(12)
        wl.addLayout(self._header())

        self.stack = QStackedWidget()
        self.stack.setStyleSheet("background-color: transparent;")
        self.stack.addWidget(self._upload_page())
        self.stack.addWidget(self._analysis_page())
        wl.addWidget(self.stack, 1)
        root.addWidget(wrapper)

    def _header(self):
        row = QHBoxLayout()
        title = QLabel("PneumoScan  //  AI DIAGNOSTIC SYSTEM")
        title.setStyleSheet(
            f"font-size: 14px; font-weight: 700; color: {HOLO_BLUE_LIGHT}; "
            f"font-family: '{FONT_FAMILY}'; letter-spacing: 3px; background-color: transparent; border: none;"
        )
        self.status_lbl = QLabel("READY")
        self._set_status_style(HOLO_GREEN, "rgba(52,168,83,0.1)", "rgba(52,168,83,0.3)")
        close_btn = QPushButton("Close")
        close_btn.setFixedSize(76, 30)
        close_btn.setCursor(QCursor(Qt.PointingHandCursor))
        close_btn.setStyleSheet(
            f"QPushButton {{ background-color: transparent; color: {HOLO_TEXT_SEC}; "
            f"border: 1px solid rgba(66,133,244,0.3); border-radius: 15px; "
            f"font-size: 10px; font-family: '{FONT_FAMILY}'; }}"
            f"QPushButton:hover {{ color: {HOLO_RED}; border-color: {HOLO_RED}; }}"
        )
        close_btn.clicked.connect(self.close)
        row.addWidget(title)
        row.addStretch()
        row.addWidget(self.status_lbl)
        row.addSpacing(10)
        row.addWidget(close_btn)
        return row

    def _upload_page(self):
        page = QWidget()
        page.setStyleSheet("background-color: transparent;")

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("background-color: transparent; border: none;")
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        inner = QWidget()
        inner.setStyleSheet("background-color: transparent;")
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(0, 0, 8, 8)
        layout.setSpacing(12)

        # Patient row
        pid_frame = GlowFrame()
        pid_row = QHBoxLayout(pid_frame)
        pid_row.setContentsMargins(18, 12, 18, 12)
        pid_row.setSpacing(12)

        pid_lbl = QLabel("PATIENT ID")
        pid_lbl.setFixedWidth(90)
        pid_lbl.setStyleSheet(
            f"font-size: 10px; color: {HOLO_TEXT_SEC}; font-family: '{FONT_FAMILY}'; letter-spacing: 1px; background-color: transparent; border: none;"
        )

        self.patient_id_field = QLineEdit()
        self.patient_id_field.setPlaceholderText("Enter patient ID (e.g. PAT-A1B2C3D4)")
        self.patient_id_field.setFixedHeight(36)
        self.patient_id_field.setStyleSheet(
            f"QLineEdit {{ background-color: rgba(15,35,70,0.85); border: 1px solid rgba(66,133,244,0.3); "
            f"border-radius: 10px; padding: 0 12px; font-size: 12px; color: {HOLO_TEXT}; "
            f"font-family: '{FONT_FAMILY}'; }}"
            f"QLineEdit:focus {{ border-color: {HOLO_BLUE}; }}"
        )

        self.patient_name_lbl = QLabel("")
        self.patient_name_lbl.setStyleSheet(
            f"font-size: 11px; color: {HOLO_BLUE_LIGHT}; font-family: '{FONT_FAMILY}'; background-color: transparent; border: none;"
        )

        doc_name = self.doctor.get("name", "Doctor")
        doc_lbl = QLabel(f"Dr. {doc_name}")
        doc_lbl.setStyleSheet(
            f"font-size: 11px; color: {HOLO_TEXT_SEC}; font-family: '{FONT_FAMILY}'; background-color: transparent; border: none;"
        )

        self.patient_id_field.textChanged.connect(self._on_patient_id_changed)

        pid_row.addWidget(pid_lbl)
        pid_row.addWidget(self.patient_id_field, 1)
        pid_row.addWidget(self.patient_name_lbl)
        pid_row.addWidget(doc_lbl)
        layout.addWidget(pid_frame)

        self.pid_error_lbl = QLabel("Patient ID not found. Please register the patient first.")
        self.pid_error_lbl.setStyleSheet(
            f"color: {HOLO_RED}; font-size: 10px; font-family: '{FONT_FAMILY}'; "
            "background-color: transparent; border: none; padding: 2px 4px;"
        )
        self.pid_error_lbl.hide()
        layout.addWidget(self.pid_error_lbl)

        # Upload zone
        upload_frame = GlowFrame()
        ul = QVBoxLayout(upload_frame)
        ul.setAlignment(Qt.AlignCenter)
        ul.setSpacing(12)
        ul.setContentsMargins(20, 20, 20, 20)

        self.preview = QLabel()
        self.preview.setMinimumSize(240, 170)
        self.preview.setMaximumSize(320, 220)
        self.preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setStyleSheet(
            f"background-color: rgba(5,10,20,0.85); border: 1px dashed rgba(66,133,244,0.38); "
            f"border-radius: 10px; color: {HOLO_TEXT_SEC}; font-size: 11px; font-family: '{FONT_FAMILY}';"
        )
        self.preview.setText("X-RAY PREVIEW\n\nNo image loaded")
        ul.addWidget(self.preview, alignment=Qt.AlignCenter)

        hint = QLabel("Upload a chest X-ray image to begin AI analysis")
        hint.setAlignment(Qt.AlignCenter)
        hint.setStyleSheet(f"font-size: 11px; color: {HOLO_TEXT_SEC}; font-family: '{FONT_FAMILY}'; background-color: transparent; border: none;")
        ul.addWidget(hint)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)
        btn_row.setAlignment(Qt.AlignCenter)

        self.upload_btn = QPushButton("Upload from Device")
        self.upload_btn.setFixedSize(170, 38)
        self.upload_btn.setEnabled(False)
        self.upload_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self.upload_btn.setStyleSheet(
            f"QPushButton {{ background-color: {HOLO_BLUE}; color: white; border: none; "
            f"border-radius: {BTN_RADIUS}px; font-size: {FONT_BODY}px; font-weight: 600; "
            f"font-family: '{FONT_FAMILY}'; }}"
            f"QPushButton:hover {{ background-color: {HOLO_BLUE_GLOW}; }}"
            f"QPushButton:disabled {{ background-color: rgba(66,133,244,0.2); "
            f"color: rgba(255,255,255,0.3); }}"
        )
        self.upload_btn.clicked.connect(self._upload_file)

        self.paste_btn = QPushButton("Paste from Clipboard")
        self.paste_btn.setFixedSize(170, 38)
        self.paste_btn.setEnabled(False)
        self.paste_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self.paste_btn.setStyleSheet(
            f"QPushButton {{ background-color: transparent; color: {HOLO_BLUE_LIGHT}; "
            f"border: 1px solid {HOLO_BLUE}; border-radius: {BTN_RADIUS}px; "
            f"font-size: {FONT_BODY}px; font-weight: 600; font-family: '{FONT_FAMILY}'; }}"
            f"QPushButton:hover {{ background-color: rgba(66,133,244,0.12); }}"
            f"QPushButton:disabled {{ color: rgba(137,180,250,0.25); "
            f"border-color: rgba(66,133,244,0.2); }}"
        )
        self.paste_btn.clicked.connect(self._paste_clipboard)

        btn_row.addWidget(self.upload_btn)
        btn_row.addWidget(self.paste_btn)
        ul.addLayout(btn_row)

        fmt = QLabel("Supported formats: JPEG, PNG, BMP, TIFF")
        fmt.setAlignment(Qt.AlignCenter)
        fmt.setStyleSheet(f"font-size: 9px; color: {HOLO_TEXT_SEC}; font-family: '{FONT_FAMILY}'; background-color: transparent; border: none;")
        ul.addWidget(fmt)
        layout.addWidget(upload_frame, 1)

        self.analyze_btn = QPushButton("INITIATE ANALYSIS")
        self.analyze_btn.setFixedSize(210, 44)
        self.analyze_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self.analyze_btn.setEnabled(False)
        self._style_analyze_btn(False)
        self.analyze_btn.clicked.connect(self._analyze)
        layout.addWidget(self.analyze_btn, alignment=Qt.AlignCenter)
        layout.addSpacing(4)

        scroll.setWidget(inner)
        outer = QVBoxLayout(page)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)
        return page

    def _analysis_page(self):
        page = QWidget()
        page.setStyleSheet("background-color: transparent;")
        main = QHBoxLayout(page)
        main.setContentsMargins(0, 0, 0, 0)
        main.setSpacing(14)

        # LEFT - X-ray display
        left = QVBoxLayout()
        left.setSpacing(10)
        xf = GlowFrame()
        xf.setFixedWidth(370)
        xl = QVBoxLayout(xf)
        xl.setContentsMargins(14, 14, 14, 14)
        xl.setSpacing(8)
        xl.addWidget(self._holo_lbl("X-RAY ANALYSIS"))
        self.xray_display = QLabel()
        self.xray_display.setFixedSize(342, 255)
        self.xray_display.setAlignment(Qt.AlignCenter)
        self.xray_display.setStyleSheet("background-color: rgba(5,10,20,0.95); border-radius: 8px;")
        xl.addWidget(self.xray_display)
        self.heatmap_btn = QPushButton("Show Grad-CAM Heatmap")
        self.heatmap_btn.setFixedHeight(34)
        self.heatmap_btn.setEnabled(False)
        self.heatmap_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self.heatmap_btn.setStyleSheet(
            f"QPushButton {{ background-color: transparent; color: {HOLO_BLUE_LIGHT}; "
            f"border: 1px solid rgba(66,133,244,0.35); border-radius: 17px; "
            f"font-size: 10px; font-family: '{FONT_FAMILY}'; }}"
            f"QPushButton:hover {{ background-color: rgba(66,133,244,0.1); }}"
            f"QPushButton:disabled {{ color: rgba(137,180,250,0.25); }}"
        )
        self.heatmap_btn.clicked.connect(self._toggle_heatmap)
        xl.addWidget(self.heatmap_btn)
        left.addWidget(xf)
        left.addStretch()
        main.addLayout(left)

        # CENTER
        center = QVBoxLayout()
        center.setSpacing(10)

        rf = GlowFrame()
        rl = QVBoxLayout(rf)
        rl.setContentsMargins(18, 14, 18, 14)
        rl.setSpacing(8)
        rl.addWidget(self._holo_lbl("DIAGNOSTIC RESULT"))
        top_r = QHBoxLayout()
        self.spinner = SpinnerWidget()
        top_r.addWidget(self.spinner)
        rr = QVBoxLayout()
        self.result_lbl = QLabel("ANALYZING...")
        self.result_lbl.setStyleSheet(
            f"font-size: 16px; font-weight: 700; color: {HOLO_BLUE_LIGHT}; "
            f"font-family: '{FONT_FAMILY}'; letter-spacing: 2px; background-color: transparent; border: none;"
        )
        self.conf_lbl = QLabel("Confidence: --")
        self.conf_lbl.setStyleSheet(f"font-size: 11px; color: {HOLO_TEXT_SEC}; font-family: '{FONT_FAMILY}'; background-color: transparent; border: none;")
        rr.addWidget(self.result_lbl)
        rr.addWidget(self.conf_lbl)
        rr.addStretch()
        top_r.addLayout(rr, 1)
        rl.addLayout(top_r)

        detail = QHBoxLayout()
        detail.setSpacing(10)
        sf = QFrame()
        sf.setStyleSheet("background-color: rgba(66,133,244,0.07); border-radius: 8px;")
        sl = QVBoxLayout(sf)
        sl.setContentsMargins(10, 8, 10, 8)
        sev_lbl = QLabel("SEVERITY")
        sev_lbl.setStyleSheet(f"font-size: 8px; color: {HOLO_TEXT_SEC}; font-family: '{FONT_FAMILY}'; letter-spacing: 1px; background-color: transparent; border: none;")
        self.sev_val = QLabel("--")
        self.sev_val.setStyleSheet(f"font-size: 14px; font-weight: 700; color: {HOLO_TEXT}; font-family: '{FONT_FAMILY}'; background-color: transparent; border: none;")
        sl.addWidget(sev_lbl)
        sl.addWidget(self.sev_val)

        sbf = QFrame()
        sbf.setStyleSheet("background-color: rgba(66,133,244,0.07); border-radius: 8px;")
        sbl = QVBoxLayout(sbf)
        sbl.setContentsMargins(10, 8, 10, 8)
        sub_lbl = QLabel("SUBTYPE")
        sub_lbl.setStyleSheet(f"font-size: 8px; color: {HOLO_TEXT_SEC}; font-family: '{FONT_FAMILY}'; letter-spacing: 1px; background-color: transparent; border: none;")
        self.sub_val = QLabel("--")
        self.sub_val.setStyleSheet(f"font-size: 14px; font-weight: 700; color: {HOLO_TEXT}; font-family: '{FONT_FAMILY}'; background-color: transparent; border: none;")
        sbl.addWidget(sub_lbl)
        sbl.addWidget(self.sub_val)
        detail.addWidget(sf, 1)
        detail.addWidget(sbf, 1)
        rl.addLayout(detail)
        center.addWidget(rf)

        pf = GlowFrame()
        pl = QVBoxLayout(pf)
        pl.setContentsMargins(14, 10, 14, 10)
        pl.setSpacing(5)
        pl.addWidget(self._holo_lbl("ENSEMBLE PROGRESS"))
        self.overall_bar = QProgressBar()
        self.overall_bar.setRange(0, self._total_models)
        self.overall_bar.setValue(0)
        self.overall_bar.setTextVisible(False)
        self.overall_bar.setFixedHeight(7)
        self.overall_bar.setStyleSheet(
            "QProgressBar { border: none; border-radius: 3px; background-color: rgba(66,133,244,0.12); }"
            "QProgressBar::chunk { background-color: #4285F4; border-radius: 3px; }"
        )
        self.prog_lbl = QLabel("0 / 6 models complete")
        self.prog_lbl.setStyleSheet(f"font-size: 9px; color: {HOLO_TEXT_SEC}; font-family: '{FONT_FAMILY}'; background-color: transparent; border: none;")
        pl.addWidget(self.overall_bar)
        pl.addWidget(self.prog_lbl)
        center.addWidget(pf)

        vf = GlowFrame()
        vl = QVBoxLayout(vf)
        vl.setContentsMargins(14, 14, 14, 14)
        vl.setSpacing(2)
        vl.addWidget(self._holo_lbl("MODEL VOTES"))
        model_names = {
            'mobilenetv3':    'MobileNetV3',
            'resnet50':       'ResNet-50',
            'densenet121':    'DenseNet-121',
            'efficientnet_b4':'EfficientNet-B4',
            'vit_b16':        'ViT-B/16',
            'inception_v3':   'InceptionV3',
        }
        for key, name in model_names.items():
            bar = ModelVoteBar(name)
            self._model_bars[key] = bar
            vl.addWidget(bar)
        center.addWidget(vf, 1)
        main.addLayout(center, 1)

        # RIGHT
        right = QVBoxLayout()
        right.setSpacing(10)

        cf = GlowFrame()
        cf.setFixedWidth(260)
        cl = QVBoxLayout(cf)
        cl.setContentsMargins(14, 14, 14, 14)
        cl.setSpacing(8)
        cl.addWidget(self._holo_lbl("ANALYTICS"))
        self.chart = AnalyticsChart()
        self.chart.setMinimumHeight(160)
        cl.addWidget(self.chart)
        cl.addWidget(self._holo_lbl("PNEUMONIA PROBABILITY"))
        self.prob_bar = QProgressBar()
        self.prob_bar.setRange(0, 100)
        self.prob_bar.setValue(0)
        self.prob_bar.setTextVisible(False)
        self.prob_bar.setFixedHeight(9)
        self.prob_bar.setStyleSheet(
            "QProgressBar { border: none; border-radius: 4px; background-color: rgba(66,133,244,0.12); }"
            "QProgressBar::chunk { background-color: #4285F4; border-radius: 4px; }"
        )
        self.prob_pct = QLabel("0%")
        self.prob_pct.setAlignment(Qt.AlignCenter)
        self.prob_pct.setStyleSheet(
            f"font-size: 20px; font-weight: 700; color: {HOLO_BLUE_LIGHT}; font-family: '{FONT_FAMILY}'; background-color: transparent; border: none;"
        )
        cl.addWidget(self.prob_bar)
        cl.addWidget(self.prob_pct)
        cl.addStretch()
        right.addWidget(cf)

        af = GlowFrame()
        af.setFixedWidth(260)
        al = QVBoxLayout(af)
        al.setContentsMargins(14, 14, 14, 14)
        al.setSpacing(8)
        al.addWidget(self._holo_lbl("ACTIONS"))
        self.report_btn = QPushButton("Generate PDF Report")
        self.report_btn.setFixedHeight(36)
        self.report_btn.setEnabled(False)
        self.report_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self.report_btn.setStyleSheet(
            f"QPushButton {{ background-color: transparent; color: {HOLO_BLUE_LIGHT}; "
            f"border: 1px solid {HOLO_BLUE}; border-radius: {BTN_RADIUS}px; "
            f"font-size: {FONT_BODY}px; font-weight: 600; font-family: '{FONT_FAMILY}'; }}"
            f"QPushButton:hover {{ background-color: rgba(66,133,244,0.12); }}"
            f"QPushButton:disabled {{ color: rgba(137,180,250,0.25); border-color: rgba(66,133,244,0.15); }}"
        )
        self.report_btn.clicked.connect(self._generate_report)
        new_btn = QPushButton("New Scan")
        new_btn.setFixedHeight(36)
        new_btn.setCursor(QCursor(Qt.PointingHandCursor))
        new_btn.setStyleSheet(
            f"QPushButton {{ background-color: {HOLO_BLUE}; color: white; border: none; "
            f"border-radius: {BTN_RADIUS}px; font-size: {FONT_BODY}px; font-weight: 600; "
            f"font-family: '{FONT_FAMILY}'; }}"
            f"QPushButton:hover {{ background-color: {HOLO_BLUE_GLOW}; }}"
        )
        new_btn.clicked.connect(self._new_scan)
        al.addWidget(self.report_btn)
        al.addWidget(new_btn)
        right.addWidget(af)
        right.addStretch()
        main.addLayout(right)
        return page

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _holo_lbl(self, text):
        lbl = QLabel(text)
        lbl.setStyleSheet(
            f"font-size: 10px; color: {HOLO_TEXT_SEC}; font-family: '{FONT_FAMILY}'; "
            f"letter-spacing: 1px; margin-bottom: 4px; background-color: transparent; border: none;"
        )
        return lbl

    def _style_analyze_btn(self, enabled):
        if enabled:
            self.analyze_btn.setStyleSheet(
                f"QPushButton {{ background-color: {HOLO_BLUE}; color: white; border: none; "
                f"border-radius: {BTN_RADIUS}px; font-size: 13px; font-weight: 700; "
                f"font-family: '{FONT_FAMILY}'; letter-spacing: 2px; }}"
                f"QPushButton:hover {{ background-color: {HOLO_BLUE_GLOW}; }}"
            )
        else:
            self.analyze_btn.setStyleSheet(
                f"QPushButton {{ background-color: rgba(66,133,244,0.25); color: rgba(137,180,250,0.4); "
                f"border: 1px solid rgba(66,133,244,0.15); border-radius: {BTN_RADIUS}px; "
                f"font-size: 13px; font-weight: 700; font-family: '{FONT_FAMILY}'; letter-spacing: 2px; }}"
            )

    def _set_status_style(self, fg, bg, border):
        self.status_lbl.setStyleSheet(
            f"font-size: 10px; color: {fg}; font-family: '{FONT_FAMILY}'; letter-spacing: 2px; "
            f"background-color: {bg}; border: 1px solid {border}; border-radius: 11px; padding: 3px 10px;"
        )

    def _on_patient_id_changed(self, text):
        """Live lookup: validate patient ID and gate upload controls."""
        text = text.strip()
        if not text:
            self.patient_name_lbl.setText("")
            self.patient = None
            self.pid_error_lbl.hide()
            self._set_upload_enabled(False)
            return
        try:
            found = db.get_patient_by_id(text)
            if found:
                self.patient = found
                self.patient_name_lbl.setText(found.get("name", ""))
                self.pid_error_lbl.hide()
                self._set_upload_enabled(True)
            else:
                self.patient = None
                self.patient_name_lbl.setText("")
                self.pid_error_lbl.show()
                self._set_upload_enabled(False)
        except Exception:
            pass

    def _set_upload_enabled(self, enabled: bool):
        self.upload_btn.setEnabled(enabled)
        self.paste_btn.setEnabled(enabled)

    # ------------------------------------------------------------------
    # File handling
    # ------------------------------------------------------------------

    def _upload_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Upload X-Ray", "",
            "Images (*.jpg *.jpeg *.png *.bmp *.tiff *.tif);;All Files (*)"
        )
        if path:
            self._load_image(path)

    def _paste_clipboard(self):
        clipboard = QApplication.clipboard()
        pixmap = clipboard.pixmap()
        if not pixmap.isNull():
            os.makedirs("outputs", exist_ok=True)
            path = "outputs/temp_paste.png"
            pixmap.save(path)
            self._load_image(path)
        else:
            text = clipboard.text().strip()
            if text and os.path.exists(text):
                self._load_image(text)

    def _load_image(self, path):
        self.image_path = path
        pm = QPixmap(path)
        if pm.isNull():
            return
        self.preview.setPixmap(pm.scaled(300, 210, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.preview.setText("")
        self.analyze_btn.setEnabled(True)
        self._style_analyze_btn(True)

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def _analyze(self):
        if not self.image_path:
            return

        # Validate patient exists before running analysis
        pid = self.patient_id_field.text().strip()
        if pid and not self.patient:
            QMessageBox.warning(
                self, "Patient Not Found",
                f"No patient found with ID '{pid}'.\n"
                "Please register the patient first or leave the field blank."
            )
            return

        self.stack.setCurrentIndex(1)
        pm = QPixmap(self.image_path)
        if not pm.isNull():
            self.xray_display.setPixmap(pm.scaled(342, 255, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.spinner.start()
        self.status_lbl.setText("ANALYZING")
        self._set_status_style(HOLO_BLUE_LIGHT, "rgba(66,133,244,0.1)", "rgba(66,133,244,0.3)")
        self._models_done = 0
        self.overall_bar.setValue(0)
        self.prog_lbl.setText(f"0 / {self._total_models} models complete")

        os.makedirs("outputs/heatmaps", exist_ok=True)

        self._thread = QThread()
        self._worker = InferenceWorker(self.engine, self.image_path, "outputs/heatmaps")
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.quick_done.connect(self._on_quick)
        self._worker.model_done.connect(self._on_model)
        self._worker.full_done.connect(self._on_full)
        self._worker.error.connect(self._on_error)
        self._thread.start()

    def _on_quick(self, r):
        pred = r.get('prediction', '')
        conf = r.get('confidence', 0)
        prob = r.get('probability', conf)
        self._update_result_label(pred, conf)
        if 'mobilenetv3' in self._model_bars:
            self._model_bars['mobilenetv3'].set_value(prob)
            self._models_done = min(self._models_done + 1, self._total_models)
            self.overall_bar.setValue(self._models_done)
            self.prog_lbl.setText(f"{self._models_done} / {self._total_models} models complete")

    def _on_model(self, name, prob):
        if name in self._model_bars:
            self._model_bars[name].set_value(prob)
        if name != 'mobilenetv3':
            self._models_done = min(self._models_done + 1, self._total_models)
            self.overall_bar.setValue(self._models_done)
            self.prog_lbl.setText(f"{self._models_done} / {self._total_models} models complete")

    def _on_full(self, r):
        self.result = r
        self.spinner.stop()

        pred = r.get('prediction', '')
        conf = r.get('confidence', 0)
        prob = r.get('ensemble_prob', 0)

        self._update_result_label(pred, conf)
        self.sev_val.setText(r.get('severity', '--'))
        self.sub_val.setText(r.get('subtype') or '--')
        self.overall_bar.setValue(self._total_models)
        self.prog_lbl.setText("Analysis complete")
        self.prob_bar.setValue(int(prob * 100))
        self.prob_pct.setText(f"{prob*100:.1f}%")
        self.chart.set_data(r.get('model_votes', {}))

        hp = r.get('heatmap_path')
        if hp and os.path.exists(hp):
            self._heatmap_path = hp
            self.heatmap_btn.setEnabled(True)

        self.report_btn.setEnabled(True)
        self.status_lbl.setText("COMPLETE")
        self._set_status_style(HOLO_GREEN, "rgba(52,168,83,0.1)", "rgba(52,168,83,0.3)")

        if self._thread:
            self._thread.quit()

        # Save to MongoDB if patient is selected
        self._save_to_db(r)

    def _save_to_db(self, r: dict):
        """Save completed scan to MongoDB with transaction and audit log."""
        if not self.patient:
            return

        patient_id = self.patient.get("patient_id", "")
        doctor_id  = self.doctor.get("doctor_id", "")

        if not patient_id or not doctor_id:
            return

        try:
            result_dict = {
                "prediction":   r.get("prediction", "Unknown"),
                "confidence":   r.get("confidence", 0.0),
                "ensemble_prob":r.get("ensemble_prob", 0.0),
                "severity":     r.get("severity", "None"),
                "subtype":      r.get("subtype", "N/A"),
            }
            model_votes = r.get("model_votes", {})
            # ensure attention_cnn key exists
            if "attention_cnn" not in model_votes:
                model_votes["attention_cnn"] = 0.0

            saved = db.save_scan(
                patient_id=patient_id,
                doctor_id=doctor_id,
                image_path=self.image_path or "",
                heatmap_path=self._heatmap_path or "",
                report_path="",
                result=result_dict,
                model_votes=model_votes,
                doctor_notes="",
                processing_time_ms=r.get("processing_time_ms", 0),
            )
            self._saved_scan_id = saved.get("scan_id")
        except Exception as e:
            print(f"Warning: could not save scan to MongoDB: {e}")

    def _on_error(self, msg):
        self.spinner.stop()
        self.result_lbl.setText("ERROR")
        self.conf_lbl.setText(f"Error: {msg[:60]}")
        self.status_lbl.setText("ERROR")
        self._set_status_style(HOLO_RED, "rgba(234,67,53,0.1)", "rgba(234,67,53,0.3)")
        if self._thread:
            self._thread.quit()

    def _update_result_label(self, pred, conf):
        self.result_lbl.setText(pred)
        color = HOLO_RED if pred == 'PNEUMONIA' else HOLO_GREEN
        self.result_lbl.setStyleSheet(
            f"font-size: 16px; font-weight: 700; color: {color}; "
            f"font-family: '{FONT_FAMILY}'; letter-spacing: 2px; background-color: transparent; border: none;"
        )
        self.conf_lbl.setText(f"Confidence: {conf*100:.2f}%")

    def _toggle_heatmap(self):
        if not self._heatmap_path:
            return
        if self._showing_heatmap:
            pm = QPixmap(self.image_path)
            self.heatmap_btn.setText("Show Grad-CAM Heatmap")
        else:
            pm = QPixmap(self._heatmap_path)
            self.heatmap_btn.setText("Show Original X-Ray")
        if not pm.isNull():
            self.xray_display.setPixmap(pm.scaled(342, 255, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self._showing_heatmap = not self._showing_heatmap

    def _new_scan(self):
        self.image_path = None
        self.result = None
        self._heatmap_path = None
        self._showing_heatmap = False
        self._models_done = 0
        self._saved_scan_id = None

        self.pid_error_lbl.hide()
        if not (self.patient and self.patient_id_field.isReadOnly()):
            self.patient = None
            self.patient_id_field.clear()
            self.patient_name_lbl.setText("")
            self._set_upload_enabled(False)
        # if patient was pre-filled (read-only), upload stays enabled

        self.preview.clear()
        self.preview.setText("X-RAY PREVIEW\n\nNo image loaded")
        self.analyze_btn.setEnabled(False)
        self._style_analyze_btn(False)

        for b in self._model_bars.values():
            b.reset()

        self.result_lbl.setText("ANALYZING...")
        self.result_lbl.setStyleSheet(
            f"font-size: 16px; font-weight: 700; color: {HOLO_BLUE_LIGHT}; "
            f"font-family: '{FONT_FAMILY}'; letter-spacing: 2px; background-color: transparent; border: none;"
        )
        self.conf_lbl.setText("Confidence: --")
        self.sev_val.setText("--")
        self.sub_val.setText("--")
        self.prob_bar.setValue(0)
        self.prob_pct.setText("0%")
        self.overall_bar.setValue(0)
        self.prog_lbl.setText(f"0 / {self._total_models} models complete")
        self.report_btn.setEnabled(False)
        self.heatmap_btn.setEnabled(False)
        self.heatmap_btn.setText("Show Grad-CAM Heatmap")
        self.status_lbl.setText("READY")
        self._set_status_style(HOLO_GREEN, "rgba(52,168,83,0.1)", "rgba(52,168,83,0.3)")
        self.stack.setCurrentIndex(0)

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def _generate_report(self):
        if not self.result:
            return

        # Ask for doctor notes via a small inline dialog
        notes_dlg = _NotesDialog(self)
        if notes_dlg.exec() != QDialog.Accepted:
            return

        doctor_notes = notes_dlg.get_notes()

        self.report_btn.setText("Generating...")
        self.report_btn.setEnabled(False)

        try:
            patient = self.patient or {}
            doctor  = self.doctor  or {}
            r       = self.result

            result_dict = {
                "prediction":    r.get("prediction",   "Unknown"),
                "confidence":    r.get("confidence",   0.0),
                "ensemble_prob": r.get("ensemble_prob",0.0),
                "severity":      r.get("severity",     "None"),
                "subtype":       r.get("subtype",      "N/A"),
            }
            votes = dict(r.get("model_votes", {}))
            votes.setdefault("attention_cnn", 0.0)

            from datetime import datetime, timezone
            scan_date = datetime.now(timezone.utc)

            pdf_path = generate_report(
                patient      = patient,
                doctor       = doctor,
                result       = result_dict,
                model_votes  = votes,
                scan_date    = scan_date,
                heatmap_path = self._heatmap_path or "",
                doctor_notes = doctor_notes,
                output_dir   = "outputs/reports",
            )

            # Persist report path in MongoDB if scan was saved
            if self._saved_scan_id:
                try:
                    db.update_scan_report_path(self._saved_scan_id, pdf_path)
                except Exception as e:
                    print(f"Warning: could not update report path in DB: {e}")

            self._show_report_success(pdf_path)

        except Exception as e:
            QMessageBox.critical(
                self, "Report Error",
                f"Could not generate report:\n{e}",
            )
        finally:
            self.report_btn.setText("Generate PDF Report")
            self.report_btn.setEnabled(True)

    def _show_report_success(self, pdf_path: str):
        dlg = QDialog(self)
        dlg.setWindowTitle("Report Generated")
        dlg.setFixedWidth(460)
        dlg.setModal(True)
        dlg.setStyleSheet(
            f"background-color: rgba(10,25,50,0.98); "
            f"border: 1px solid rgba(66,133,244,0.5); border-radius: 12px;"
        )

        layout = QVBoxLayout(dlg)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(12)

        title = QLabel("Report Generated Successfully")
        title.setStyleSheet(
            f"font-size: 13px; font-weight: 700; color: {HOLO_GREEN}; "
            f"font-family: '{FONT_FAMILY}'; background-color: transparent; border: none;"
        )
        layout.addWidget(title)

        msg_lbl = QLabel("PDF report saved to:")
        msg_lbl.setStyleSheet(
            f"font-size: 11px; color: {HOLO_TEXT}; font-family: '{FONT_FAMILY}'; "
            "background-color: transparent; border: none;"
        )
        layout.addWidget(msg_lbl)

        path_lbl = QLabel(pdf_path)
        path_lbl.setWordWrap(True)
        path_lbl.setStyleSheet(
            f"font-size: 10px; color: {HOLO_BLUE_LIGHT}; font-family: '{FONT_FAMILY}'; "
            f"background-color: rgba(15,35,70,0.9); border: 1px solid rgba(66,133,244,0.4); "
            f"border-radius: 8px; padding: 8px 10px;"
        )
        layout.addWidget(path_lbl)

        btn_row = QHBoxLayout()
        ok_btn = QPushButton("OK")
        ok_btn.setFixedHeight(36)
        ok_btn.setCursor(QCursor(Qt.PointingHandCursor))
        ok_btn.setStyleSheet(
            f"QPushButton {{ background-color: transparent; color: {HOLO_TEXT_SEC}; "
            f"border: 1px solid rgba(66,133,244,0.3); border-radius: {BTN_RADIUS}px; "
            f"font-size: {FONT_BODY}px; font-family: '{FONT_FAMILY}'; }}"
            f"QPushButton:hover {{ color: {HOLO_BLUE_LIGHT}; }}"
        )
        ok_btn.clicked.connect(dlg.accept)

        open_btn = QPushButton("Open File")
        open_btn.setFixedHeight(36)
        open_btn.setCursor(QCursor(Qt.PointingHandCursor))
        open_btn.setStyleSheet(
            f"QPushButton {{ background-color: {HOLO_BLUE}; color: white; border: none; "
            f"border-radius: {BTN_RADIUS}px; font-size: {FONT_BODY}px; font-weight: 600; "
            f"font-family: '{FONT_FAMILY}'; }}"
            f"QPushButton:hover {{ background-color: {HOLO_BLUE_GLOW}; }}"
        )
        _open = [False]
        def _on_open():
            _open[0] = True
            dlg.accept()
        open_btn.clicked.connect(_on_open)

        btn_row.addWidget(ok_btn)
        btn_row.addStretch()
        btn_row.addWidget(open_btn)
        layout.addLayout(btn_row)

        dlg.exec()

        if _open[0]:
            import subprocess, sys as _sys
            if _sys.platform.startswith("win"):
                os.startfile(pdf_path)
            elif _sys.platform == "darwin":
                subprocess.call(["open", pdf_path])
            else:
                subprocess.call(["xdg-open", pdf_path])

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.canvas.setGeometry(0, 0, self.width(), self.height())