"""
NEMO Scan - GUI Styles
Location: gui/styles.py

Single source of truth for all colors, fonts, dimensions, and stylesheets.
"""

BG          = "#F8F9FA"
SURFACE     = "#FFFFFF"
BORDER      = "#E8EAED"
TEXT        = "#202124"
TEXT_SEC    = "#5F6368"
BLUE        = "#4285F4"
RED         = "#EA4335"
GREEN       = "#34A853"
YELLOW      = "#FBBC05"
BLUE_TINT   = "#E8F0FE"
RED_TINT    = "#FCE8E6"
GREEN_TINT  = "#E6F4EA"
YELLOW_TINT = "#FEF7E0"

HOLO_BG         = "#050A14"
HOLO_BLUE       = "#4285F4"
HOLO_BLUE_GLOW  = "#1A73E8"
HOLO_BLUE_LIGHT = "#89B4FA"
HOLO_TEXT       = "#E8F0FE"
HOLO_TEXT_SEC   = "#8AB4F8"
HOLO_GREEN      = "#34A853"
HOLO_RED        = "#EA4335"

SIDEBAR_W    = 240
CARD_RADIUS  = 16
BTN_RADIUS   = 20
CONTENT_PAD  = 40

FONT_FAMILY  = "Segoe UI"
FONT_METRIC  = 28
FONT_HEADER  = 14
FONT_BODY    = 12
FONT_LABEL   = 10

MAIN_STYLE = f"""
* {{ font-family: '{FONT_FAMILY}'; font-size: {FONT_BODY}px; }}
QMainWindow, QWidget {{ color: {TEXT}; }}
QMainWindow {{ background-color: {BG}; }}
QWidget#main_window {{ background-color: {BG}; }}
QWidget#sidebar {{ background-color: {SURFACE}; border-right: 1px solid {BORDER}; }}
QWidget#content_area {{ background-color: {BG}; }}
QFrame#card {{
    background-color: {SURFACE}; border: 1px solid {BORDER};
    border-radius: {CARD_RADIUS}px;
}}
QFrame#card:hover {{ border-color: {BLUE}; }}
QPushButton#btn_primary {{
    background-color: {BLUE}; color: white; border: none;
    border-radius: {BTN_RADIUS}px; padding: 10px 24px;
    font-size: {FONT_BODY}px; font-weight: 600;
}}
QPushButton#btn_primary:hover {{ background-color: #1A73E8; }}
QPushButton#btn_primary:pressed {{ background-color: #1557B0; }}
QPushButton#btn_outline {{
    background-color: transparent; color: {BLUE};
    border: 1.5px solid {BLUE}; border-radius: {BTN_RADIUS}px;
    padding: 9px 24px; font-size: {FONT_BODY}px; font-weight: 600;
}}
QPushButton#btn_outline:hover {{ background-color: {BLUE_TINT}; }}
QPushButton#nav_btn {{
    background-color: transparent; color: {TEXT_SEC}; border: none;
    border-radius: 10px; padding: 10px 16px;
    font-size: {FONT_BODY}px; text-align: left;
}}
QPushButton#nav_btn:hover {{ background-color: {BLUE_TINT}; color: {BLUE}; }}
QPushButton#nav_btn[active=true] {{
    background-color: {BLUE_TINT}; color: {BLUE}; font-weight: 600;
}}
QLineEdit {{
    background-color: {SURFACE}; border: 1.5px solid {BORDER};
    border-radius: {BTN_RADIUS}px; padding: 10px 16px;
    font-size: {FONT_BODY}px; color: {TEXT};
}}
QLineEdit:focus {{ border-color: {BLUE}; }}
QScrollBar:vertical {{
    border: none; background: {BG}; width: 6px; margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {BORDER}; border-radius: 3px; min-height: 20px;
}}
QScrollBar::handle:vertical:hover {{ background: {TEXT_SEC}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QTableWidget {{
    background-color: {SURFACE}; border: 1px solid {BORDER};
    border-radius: {CARD_RADIUS}px; gridline-color: {BORDER}; outline: none;
}}
QTableWidget::item {{ padding: 12px 16px; border: none; }}
QTableWidget::item:selected {{ background-color: {BLUE_TINT}; color: {BLUE}; }}
QHeaderView::section {{
    background-color: {BG}; color: {TEXT_SEC};
    font-size: {FONT_LABEL}px; font-weight: 600;
    padding: 12px 16px; border: none; border-bottom: 1px solid {BORDER};
}}
QLabel#label_metric {{ font-size: {FONT_METRIC}px; font-weight: 700; color: {TEXT}; }}
QLabel#label_header {{ font-size: {FONT_HEADER}px; font-weight: 600; color: {TEXT}; }}
QLabel#label_secondary {{ font-size: {FONT_LABEL}px; color: {TEXT_SEC}; }}
QLabel#label_section {{ font-size: {FONT_HEADER}px; font-weight: 700; color: {TEXT}; }}
QLabel#badge_normal {{
    background-color: {GREEN_TINT}; color: {GREEN};
    border-radius: 10px; padding: 3px 10px;
    font-size: {FONT_LABEL}px; font-weight: 600;
}}
QLabel#badge_pneumonia {{
    background-color: {RED_TINT}; color: {RED};
    border-radius: 10px; padding: 3px 10px;
    font-size: {FONT_LABEL}px; font-weight: 600;
}}
QLabel#badge_review {{
    background-color: {YELLOW_TINT}; color: #B06000;
    border-radius: 10px; padding: 3px 10px;
    font-size: {FONT_LABEL}px; font-weight: 600;
}}
QLabel {{ background-color: transparent; border: none; color: {TEXT}; }}
"""

HOLO_STYLE = f"""
* {{ font-family: '{FONT_FAMILY}'; color: {HOLO_TEXT}; }}
QDialog, QWidget {{ background-color: {HOLO_BG}; }}
QLabel#holo_title {{
    font-size: 20px; font-weight: 700;
    color: {HOLO_BLUE_LIGHT}; letter-spacing: 2px;
}}
QLabel#holo_label {{
    font-size: 10px; color: {HOLO_TEXT_SEC}; letter-spacing: 0.5px;
}}
QLabel#holo_value {{
    font-size: 22px; font-weight: 700; color: {HOLO_BLUE_LIGHT};
}}
QLabel#holo_result_normal {{
    font-size: 18px; font-weight: 700;
    color: {HOLO_GREEN}; letter-spacing: 2px;
}}
QLabel#holo_result_pneumonia {{
    font-size: 18px; font-weight: 700;
    color: {HOLO_RED}; letter-spacing: 2px;
}}
QPushButton#holo_btn {{
    background-color: transparent; color: {HOLO_BLUE_LIGHT};
    border: 1px solid {HOLO_BLUE}; border-radius: {BTN_RADIUS}px;
    padding: 10px 24px; font-size: {FONT_BODY}px;
    font-weight: 600; letter-spacing: 1px;
}}
QPushButton#holo_btn:hover {{
    background-color: rgba(66,133,244,0.15); border-color: {HOLO_BLUE_LIGHT};
}}
QPushButton#holo_btn_primary {{
    background-color: {HOLO_BLUE}; color: white; border: none;
    border-radius: {BTN_RADIUS}px; padding: 10px 24px;
    font-size: {FONT_BODY}px; font-weight: 600; letter-spacing: 1px;
}}
QPushButton#holo_btn_primary:hover {{ background-color: {HOLO_BLUE_GLOW}; }}
QProgressBar {{
    border: none; border-radius: 3px;
    background-color: rgba(66,133,244,0.15); height: 6px;
}}
QProgressBar::chunk {{ background-color: {HOLO_BLUE}; border-radius: 3px; }}
QScrollBar:vertical {{
    border: none; background: transparent; width: 4px;
}}
QScrollBar::handle:vertical {{
    background: rgba(66,133,244,0.4); border-radius: 2px;
}}
QLineEdit#holo_input {{
    background-color: rgba(15,35,70,0.8);
    border: 1px solid rgba(66,133,244,0.35);
    border-radius: {BTN_RADIUS}px; padding: 10px 16px;
    font-size: {FONT_BODY}px; color: {HOLO_TEXT};
}}
QLineEdit#holo_input:focus {{ border-color: {HOLO_BLUE}; }}
"""
