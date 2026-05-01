"""
PneumoScan - Application Entry Point
Location: main.py

Run this file to start PneumoScan:
    python main.py
"""

import sys
import os
import traceback

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def exception_hook(exctype, value, tb):
    traceback.print_exception(exctype, value, tb)
    sys.exit(1)

sys.excepthook = exception_hook


from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFont

from gui.styles import MAIN_STYLE, FONT_FAMILY
from gui.login_screen import LoginScreen
from gui.main_window import MainWindow
import database.db as db


def load_engine():
    try:
        from core.inference.engine import InferenceEngine
        engine = InferenceEngine()
        weights_dir = os.path.join(ROOT, 'weights', 'lung')
        engine.load_models(weights_dir)
        return engine
    except Exception as e:
        print(f'Warning: Could not load inference engine: {e}')
        print('Running in demo mode.')
        return None


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("PneumoScan")
    app.setStyle("Fusion")
    app.setFont(QFont(FONT_FAMILY, 12))
    app.setStyleSheet(MAIN_STYLE)

    engine = load_engine()

    login = LoginScreen()
    login.resize(1000, 650)
    screen = app.primaryScreen().geometry()
    login.move((screen.width() - 1000) // 2, (screen.height() - 650) // 2)

    _window_ref = []   # keeps MainWindow alive after on_login returns

    def on_login(doctor: dict):
        try:
            login.close()
            window = MainWindow(doctor=doctor, engine=engine)
            _window_ref.append(window)
            window.show()
        except Exception:
            traceback.print_exc()

    login.login_success.connect(on_login)
    login.show()

    try:
        sys.exit(app.exec())
    except Exception:
        traceback.print_exc()


if __name__ == '__main__':
    main()