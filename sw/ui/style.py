"""Feuille de style partagée (palette Catppuccin Mocha)."""

BASE = "#181825"
SURFACE = "#1e1e2e"
BORDER = "#313244"
BORDER_HOVER = "#89b4fa"
TEXT = "#cdd6f4"
TEXT_DIM = "#a6adc8"
TEXT_FAINT = "#585b70"
ACCENT = "#89b4fa"
ACCENT_HOVER = "#b4d0fb"
GREEN = "#a6e3a1"
ORANGE = "#fab387"
RED = "#f38ba8"
MAUVE = "#cba6f7"
BUTTON_MUTED = "#45475a"
BUTTON_MUTED_HOVER = "#585b70"

DIALOG = f"""
QDialog {{ background: {BASE}; color: {TEXT}; }}
QWidget {{ color: {TEXT}; }}
QTabWidget::pane {{
    border: 1px solid {BORDER}; border-radius: 10px;
    background: {BASE}; top: -1px;
}}
QTabBar::tab {{
    background: {SURFACE}; color: {TEXT_DIM};
    border: 1px solid {BORDER}; border-bottom: none;
    border-top-left-radius: 8px; border-top-right-radius: 8px;
    padding: 8px 16px; margin-right: 2px;
}}
QTabBar::tab:selected {{ background: {BASE}; color: {ACCENT}; }}
QTabBar::tab:hover {{ color: {TEXT}; }}
QGroupBox {{
    border: 1px solid {BORDER}; border-radius: 10px;
    margin-top: 14px; padding: 18px 12px 12px 12px;
    font-weight: bold; color: {TEXT_DIM};
}}
QGroupBox::title {{
    subcontrol-origin: margin; left: 14px; padding: 0 6px; color: {ACCENT};
}}
QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox {{
    background: {SURFACE}; color: {TEXT}; border: 1px solid {BORDER};
    border-radius: 8px; padding: 7px 12px; min-height: 26px;
}}
QComboBox:hover, QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
    border-color: {BORDER_HOVER};
}}
QComboBox::drop-down {{ border: none; width: 24px; }}
QComboBox QAbstractItemView {{
    background: {SURFACE}; color: {TEXT};
    selection-background-color: {BORDER}; border: 1px solid {BORDER};
}}
QPushButton {{
    background: {ACCENT}; color: #11111b; border: none;
    border-radius: 10px; padding: 10px 24px; font-weight: bold;
}}
QPushButton:hover {{ background: {ACCENT_HOVER}; }}
QPushButton:disabled {{ background: {BUTTON_MUTED}; color: {TEXT_FAINT}; }}
QPushButton[muted="true"] {{
    background: {BUTTON_MUTED}; color: {TEXT}; padding: 7px 14px; font-weight: normal;
}}
QPushButton[muted="true"]:hover {{ background: {BUTTON_MUTED_HOVER}; }}
QLabel {{ color: {TEXT_DIM}; }}
QLabel[hint="true"] {{ color: {TEXT_FAINT}; font-size: 11px; }}
QPlainTextEdit, QTextEdit {{
    background: {SURFACE}; color: {TEXT}; border: 1px solid {BORDER};
    border-radius: 8px; padding: 8px; font-size: 12px;
}}
QPlainTextEdit:focus, QTextEdit:focus {{ border-color: {BORDER_HOVER}; }}
QCheckBox {{ color: {TEXT_DIM}; spacing: 8px; }}
QCheckBox::indicator {{
    width: 16px; height: 16px; border-radius: 4px;
    border: 1px solid {BORDER}; background: {SURFACE};
}}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; }}
QTableWidget {{
    background: {SURFACE}; color: {TEXT}; border: 1px solid {BORDER};
    border-radius: 8px; gridline-color: {BORDER};
}}
QTableWidget::item:selected {{ background: {BORDER}; }}
QHeaderView::section {{
    background: {BASE}; color: {TEXT_DIM}; border: none;
    border-bottom: 1px solid {BORDER}; padding: 6px;
}}
QProgressBar {{
    background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 8px;
    text-align: center; color: {TEXT}; min-height: 18px;
}}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 7px; }}
QScrollArea {{ border: none; background: {BASE}; }}
QScrollBar:vertical {{ background: {BASE}; width: 10px; margin: 0; }}
QScrollBar::handle:vertical {{ background: {BORDER}; border-radius: 5px; min-height: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
"""

PICKER = f"""
QDialog {{ background: transparent; }}
QWidget#card {{
    background: rgba(24,24,37,246); border: 1px solid {BORDER}; border-radius: 18px;
}}
QLabel {{ color: {TEXT_DIM}; background: transparent; }}
QLabel#title {{ color: {TEXT}; font-size: 15px; font-weight: bold; }}
QLabel#preview {{ color: {TEXT_FAINT}; font-size: 11px; }}
QLabel#hint {{ color: {TEXT_FAINT}; font-size: 11px; }}
QListWidget {{
    background: transparent; border: none; color: {TEXT}; font-size: 13px; outline: none;
}}
QListWidget::item {{ padding: 7px 10px; border-radius: 8px; }}
QListWidget::item:selected {{ background: {ACCENT}; color: #11111b; }}
QListWidget::item:hover {{ background: {BORDER}; }}
QComboBox {{
    background: {SURFACE}; color: {TEXT}; border: 1px solid {BORDER};
    border-radius: 8px; padding: 5px 10px;
}}
QComboBox QAbstractItemView {{
    background: {SURFACE}; color: {TEXT}; selection-background-color: {BORDER};
}}
"""
