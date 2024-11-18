import sys
from PySide6.QtWidgets import QApplication
from src.gui.mainwindow import MainWindow  # Import MainWindow from mainwindow.py

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
