"""
NEMO Scan - AI Models Panel
Location: gui/models_panel.py

Shows status cards for all 7 models: name, role, accuracy, weight, loaded status.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QFrame, QScrollArea, QGridLayout
)
from PySide6.QtCore import Qt

from gui.styles import (
    SURFACE, BORDER, TEXT, TEXT_SEC, BLUE, RED, GREEN, YELLOW,
    BLUE_TINT, GREEN_TINT, RED_TINT,
    BG, FONT_FAMILY, CONTENT_PAD, FONT_BODY, FONT_LABEL,
    FONT_HEADER, CARD_RADIUS
)


MODEL_INFO = [
    {
        "name":     "MobileNetV3",
        "key":      "mobilenetv3",
        "role":     "Speed Model",
        "desc":     "Instant result in 2-3 seconds. Runs first.",
        "weight":   "0.10",
        "val_acc":  "97.13%",
        "test_acc": "97.80%",
        "params":   "5.4M",
        "accent":   BLUE,
        "tint":     BLUE_TINT,
    },
    {
        "name":     "ResNet-50",
        "key":      "resnet50",
        "role":     "Core Ensemble",
        "desc":     "Strong general-purpose feature extractor.",
        "weight":   "0.18",
        "val_acc":  "98.47%",
        "test_acc": "97.67%",
        "params":   "25.6M",
        "accent":   GREEN,
        "tint":     GREEN_TINT,
    },
    {
        "name":     "DenseNet-121",
        "key":      "densenet121",
        "role":     "Anchor Model",
        "desc":     "CheXNet architecture. Highest ensemble weight.",
        "weight":   "0.25",
        "val_acc":  "98.87%",
        "test_acc": "98.67%",
        "params":   "7.0M",
        "accent":   BLUE,
        "tint":     BLUE_TINT,
    },
    {
        "name":     "EfficientNet-B4",
        "key":      "efficientnet_b4",
        "role":     "Efficiency Model",
        "desc":     "Compound scaling for accuracy-efficiency balance.",
        "weight":   "0.20",
        "val_acc":  "98.40%",
        "test_acc": "98.20%",
        "params":   "19.3M",
        "accent":   GREEN,
        "tint":     GREEN_TINT,
    },
    {
        "name":     "ViT-B/16",
        "key":      "vit_b16",
        "role":     "Attention Model",
        "desc":     "Transformer-based global pattern detection.",
        "weight":   "0.17",
        "val_acc":  "98.13%",
        "test_acc": "97.73%",
        "params":   "86.6M",
        "accent":   YELLOW,
        "tint":     "#FEF7E0",
    },
    {
        "name":     "InceptionV3",
        "key":      "inception_v3",
        "role":     "Multi-Scale Detector",
        "desc":     "Detects fine and coarse features simultaneously.",
        "weight":   "0.10",
        "val_acc":  "98.80%",
        "test_acc": "98.73%",
        "params":   "23.9M",
        "accent":   GREEN,
        "tint":     GREEN_TINT,
    },
    {
        "name":     "AttentionCNN",
        "key":      "attention_cnn",
        "role":     "Explainability Model",
        "desc":     "Custom architecture. Produces Grad-CAM heatmaps only.",
        "weight":   "0.00",
        "val_acc":  "98.67%",
        "test_acc": "98.13%",
        "params":   "~30M",
        "accent":   RED,
        "tint":     RED_TINT,
    },
]


class ModelCard(QFrame):
    def __init__(self, info: dict, loaded: bool = True, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        self.setMinimumHeight(180)
        self._build(info, loaded)

    def _build(self, info, loaded):
        accent = info["accent"]
        tint   = info["tint"]
        self.setStyleSheet(f"""
            QFrame#card {{
                background-color: {SURFACE};
                border: 1px solid {BORDER};
                border-radius: {CARD_RADIUS}px;
            }}
            QFrame#card:hover {{ border-color: {accent}; }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(8)

        # Top row: name + status badge
        top = QHBoxLayout()
        name_lbl = QLabel(info["name"])
        name_lbl.setStyleSheet(f"font-size: 14px; font-weight: 700; color: {TEXT}; font-family: '{FONT_FAMILY}';")

        status_color = GREEN if loaded else RED
        status_text  = "Loaded" if loaded else "Missing"
        status_bg    = GREEN_TINT if loaded else RED_TINT
        status_lbl = QLabel(status_text)
        status_lbl.setStyleSheet(f"""
            background-color: {status_bg};
            color: {status_color};
            border-radius: 10px;
            padding: 3px 10px;
            font-size: {FONT_LABEL}px;
            font-weight: 600;
            font-family: '{FONT_FAMILY}';
        """)

        top.addWidget(name_lbl)
        top.addStretch()
        top.addWidget(status_lbl)
        layout.addLayout(top)

        # Role pill
        role_lbl = QLabel(info["role"])
        role_lbl.setStyleSheet(f"""
            background-color: {tint};
            color: {accent};
            border-radius: 8px;
            padding: 2px 10px;
            font-size: {FONT_LABEL}px;
            font-weight: 600;
            font-family: '{FONT_FAMILY}';
        """)
        role_lbl.setFixedHeight(22)
        layout.addWidget(role_lbl, alignment=Qt.AlignLeft)

        # Description
        desc = QLabel(info["desc"])
        desc.setStyleSheet(f"font-size: {FONT_LABEL}px; color: {TEXT_SEC}; font-family: '{FONT_FAMILY}';")
        desc.setWordWrap(True)
        layout.addWidget(desc)
        layout.addStretch()

        # Stats row
        stats = QHBoxLayout()
        stats.setSpacing(16)

        for label, val in [
            ("Val Acc",    info["val_acc"]),
            ("Test Acc",   info["test_acc"]),
            ("Weight",     info["weight"]),
            ("Params",     info["params"]),
        ]:
            block = QVBoxLayout()
            block.setSpacing(2)
            v = QLabel(val)
            v.setStyleSheet(f"font-size: 13px; font-weight: 700; color: {TEXT}; font-family: '{FONT_FAMILY}';")
            l = QLabel(label)
            l.setStyleSheet(f"font-size: {FONT_LABEL}px; color: {TEXT_SEC}; font-family: '{FONT_FAMILY}';")
            block.addWidget(v)
            block.addWidget(l)
            stats.addLayout(block)

        stats.addStretch()
        layout.addLayout(stats)


class ModelsPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background-color: {BG};")
        self._build_ui()

    def _build_ui(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("background-color: transparent; border: none;")

        content = QWidget()
        content.setStyleSheet(f"background-color: {BG};")
        layout = QVBoxLayout(content)
        layout.setContentsMargins(CONTENT_PAD, CONTENT_PAD, CONTENT_PAD, CONTENT_PAD)
        layout.setSpacing(0)

        header = QLabel("AI Models")
        header.setStyleSheet(f"font-size: 24px; font-weight: 700; color: {TEXT}; font-family: '{FONT_FAMILY}';")
        layout.addWidget(header)
        layout.addSpacing(4)
        sub = QLabel("Status and performance metrics for all 7 ensemble models.")
        sub.setStyleSheet(f"font-size: {FONT_BODY}px; color: {TEXT_SEC}; font-family: '{FONT_FAMILY}';")
        layout.addWidget(sub)
        layout.addSpacing(8)

        # Summary bar
        summary = QFrame()
        summary.setStyleSheet(f"background-color: {SURFACE}; border: 1px solid {BORDER}; border-radius: 12px;")
        sl = QHBoxLayout(summary)
        sl.setContentsMargins(20, 14, 20, 14)
        sl.setSpacing(32)
        for label, val in [("Models Loaded", "7 / 7"), ("Ensemble Acc.", "98.4%"), ("Total Parameters", "~198M"), ("Architecture", "Weighted Soft-Vote")]:
            b = QVBoxLayout(); b.setSpacing(2)
            v = QLabel(val); v.setStyleSheet(f"font-size: 16px; font-weight: 700; color: {TEXT}; font-family: '{FONT_FAMILY}';")
            l = QLabel(label); l.setStyleSheet(f"font-size: {FONT_LABEL}px; color: {TEXT_SEC}; font-family: '{FONT_FAMILY}';")
            b.addWidget(v); b.addWidget(l)
            sl.addLayout(b)
        sl.addStretch()
        layout.addWidget(summary)
        layout.addSpacing(24)

        # Model cards grid
        grid = QGridLayout()
        grid.setSpacing(16)
        for i, info in enumerate(MODEL_INFO):
            card = ModelCard(info, loaded=True)
            grid.addWidget(card, i // 2, i % 2)
        layout.addLayout(grid)
        layout.addStretch()

        scroll.setWidget(content)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)
