from PySide6.QtWidgets import QMainWindow, QPushButton


class MainWindow(QMainWindow):
    """
    Main window class.
    """

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Seam Carving")
