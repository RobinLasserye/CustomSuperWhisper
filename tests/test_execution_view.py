import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication
from sw.pipeline import Execution, Step
from sw.ui.execution import ExecutionDialog


def test_select_step_shows_plain_text_and_clear_discards_history():
    app = QApplication.instance() or QApplication([])
    dialog = ExecutionDialog()
    report = Execution("langgraph", "<b>raw</b>", "message", "en", "ollama", "local")
    report.output = "after"
    report.steps = [Step("format", 12.5, report.source, "after", "→ validate", 1)]
    dialog.add_execution(report)
    assert dialog.steps.rowCount() == 1
    assert dialog.before.toPlainText() == "<b>raw</b>"
    assert dialog.after.toPlainText() == "after"
    for _ in range(12):
        dialog.add_execution(report)
    assert dialog.runs.count() == 10
    dialog.clear_history()
    assert dialog.runs.count() == 0
    assert dialog.before.toPlainText() == dialog.after.toPlainText() == ""
    assert not dialog.reports
    dialog.close()
