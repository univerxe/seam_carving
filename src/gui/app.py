import sys
import traceback

from PySide6.QtWidgets import QApplication, QMessageBox

from src.gui.mainwindow import MainWindow


class Application(QApplication):
    """
    Root class for the application window.
    """

    def __init__(self):
        super().__init__(sys.argv)
        self.window = MainWindow()

        setattr(sys, "excepthook", self.handle_exception)

    def handle_exception(self, exctype, value: BaseException, tb):
        """
        Show a critical messagebox with the exception details.

        Args:
            exctype (type): The type of the exception.
            value (BaseException): The exception instance.
            tb (types.TracebackType): The traceback object.
        """
        msg = "An unhandled exception occurred.\n"
        msg += f"{exctype.__name__}("
        msg += f'"{value}")\n'
        # msg += f"File: {tb.tb_frame.f_code.co_filename}\n"
        # msg += f"Line: {tb.tb_lineno}\n"
        msg += "=== | Traceback | ===\n"
        msg += "".join(traceback.format_tb(tb))

        QMessageBox.critical(self.window, "Error", msg)

    def run(self) -> int:
        """
        Run the application and show the main window.

        Returns:
            int: Exit code.
        """
        self.window.show()
        return self.exec_()
