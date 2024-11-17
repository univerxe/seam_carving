import sys
from PySide6.QtWidgets import (
    QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QLabel,
    QFileDialog, QLineEdit, QApplication, QGroupBox, QComboBox
)
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt
from src.lib import Image, CarvableImage
from src.algorithms.energy import EnergyCalculator
from src.algorithms.seam import SeamFinder
import cv2

class AppStyles:
    """
    Centralized styles for the application.
    """
    WINDOW_STYLE = """
        QMainWindow {
            background-color: #2C2C2C;
        }
    """
    BUTTON_STYLE = """
        QPushButton {
            background-color: #0C8CE9;
            color: white;
            border-radius: 5px;
            padding: 10px 20px; /* Adjust padding for button size */
            font-size: 16px;   /* Dynamic font size for buttons */
        }
        QPushButton:hover {
            background-color: #005A9E;
        }
    """
    
    DROP_DOWN_STYLE = """
        QComboBox {
            background-color: white;
            color: black;
            border: 1px solid gray;
            border-radius: 5px;
            padding: 10px;
            font-size: 16px;
            min-width: 150px; /* Minimum width of the input field */
        }
        
        QComboBox::drop-down {
            border: 0px;
            background-color: white;
            width: 30px;
            border-top-right-radius: 5px;
            border-bottom-right-radius: 5px;
            
        }
        
        QComboBox::down-arrow {
            image: url("X:/2024-2/Projects/Seam Carving Dev/seam_carving/src/gui/assets/down_arrow.png");
            
            width: 20px;
            height: 20px;
        }
        
        QComboBox QAbstractItemView {
            background-color: white; /* Dropdown background color */
            color: black;
            border: 1px solid gray;
            selection-background-color: lightgray; /* Highlight color when selecting an option */
            font-size: 14px;
        }   
    """
    
    LINE_EDIT_STYLE = """
        QLineEdit {
            background-color: white;
            color: black;
            border: 1px solid gray;
            border-radius: 5px;
            padding: 10px;
            font-size: 16px;
            min-width: 150px; /* Minimum width of the input field */
        }
    """
    IMAGE_LABEL_STYLE = """
        QLabel {
            border: 2px solid gray;
            background-color: white;
            color: gray;
            font-size: 18px;
        }
    """
    GROUP_BOX_STYLE = """
        QGroupBox {
            font-weight: bold;
            border: 1px solid #ccc;
            border-radius: 5px;
            padding: 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0 10px;
        }
    """


class MainWindow(QMainWindow):
    """
    Main window class for interactive seam carving with centralized styling.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Seam Carving")
        self.setGeometry(100, 100, 1000, 800)
        self.setStyleSheet(AppStyles.WINDOW_STYLE)

        self.original_image = None
        self.carved_image = None
        self.export_button = None

        self._setup_ui()

    def _setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        # Add Controls
        controls_group = self._create_controls_group()
        main_layout.addWidget(controls_group, alignment=Qt.AlignCenter)

        # Add Image Display
        image_display_group = self._create_image_display_group()
        main_layout.addWidget(image_display_group)
        
        # Add Export Button
        export_layout = QHBoxLayout()
        export_layout.setAlignment(Qt.AlignCenter)
         
        self.export_button = QPushButton("Export Image")
        self.export_button.setStyleSheet(AppStyles.BUTTON_STYLE)
        self.export_button.clicked.connect(self.export_image)
        export_layout.addWidget(self.export_button)
        
        main_layout.addLayout(export_layout)

    def _create_controls_group(self):
        controls_group = QGroupBox("Image Controls")
        controls_group.setStyleSheet(AppStyles.GROUP_BOX_STYLE)
        controls_layout = QVBoxLayout()
        controls_group.setLayout(controls_layout)

        button_layout = QHBoxLayout()
        button_layout.setAlignment(Qt.AlignCenter)

        # Load Button
        self.load_button = QPushButton("Load Image")
        self.load_button.setStyleSheet(AppStyles.BUTTON_STYLE)
        self.load_button.clicked.connect(self.load_image)

        # Aspect Ratio Dropdown
        self.aspect_ratio_dropdown = QComboBox()
        self.aspect_ratio_dropdown.addItems(["16:9", "4:5 (Instagram Post)", "1:1", "3:4", "9:16", "Custom"])
        self.aspect_ratio_dropdown.setStyleSheet(AppStyles.DROP_DOWN_STYLE)
        
        # Seam Input
        self.seams_input = QLineEdit()
        self.seams_input.setPlaceholderText("Enter number of seams to carve")
        self.seams_input.setStyleSheet(AppStyles.LINE_EDIT_STYLE)

        # Carve Button
        self.carve_button = QPushButton("Resize Image")
        self.carve_button.setStyleSheet(AppStyles.BUTTON_STYLE)
        self.carve_button.clicked.connect(self.st_seam_carving)

        # Add to layout
        button_layout.addWidget(self.load_button)
        button_layout.addWidget(self.aspect_ratio_dropdown)
        button_layout.addWidget(self.seams_input)
        button_layout.addWidget(self.carve_button)

        controls_layout.addLayout(button_layout)
        return controls_group

    def _create_image_display_group(self):
        image_display_group = QGroupBox("Image Comparison")
        image_display_group.setStyleSheet(AppStyles.GROUP_BOX_STYLE)
        image_layout = QHBoxLayout()
        image_display_group.setLayout(image_layout)

        self.original_image_label = self._create_image_label("Before (Original Image)")
        self.carved_image_label = self._create_image_label("After (Carved Image)")

        image_layout.addWidget(self.original_image_label)
        image_layout.addWidget(self.carved_image_label)
        return image_display_group

    def _create_image_label(self, text):
        layout = QVBoxLayout()
        label_text = QLabel(text)
        label_text.setAlignment(Qt.AlignCenter)

        image_label = QLabel()
        image_label.setAlignment(Qt.AlignCenter)
        image_label.setStyleSheet(AppStyles.IMAGE_LABEL_STYLE)
        image_label.setText("No Image Loaded")  # Placeholder text
        # image_label.setMinimumSize(700, 500)

        layout.addWidget(label_text)
        layout.addWidget(image_label)
        return image_label

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.xpm *.jpg *.bmp)")
        if file_path:
            self.original_image = Image.from_path(file_path)
            self._display_image(self.original_image.mat, self.original_image_label)
            
    def export_image(self):
        if not self.carved_image:
            return

        file_path, _ = QFileDialog.getSaveFileName(self, "Save Image as", "", "Images (*.png *.xpm *.jpg *.bmp)")
        if file_path:
            self.carved_image.save(file_path)
            

    def st_seam_carving(self):
        if not self.original_image:
            return

        try:
            num_v_seams = int(self.seams_input.text())
            num_h_seams = int(self.seams_input.text())
            
            if num_v_seams < 1 or num_h_seams < 1:
                return

        except ValueError:
            return

        # Vertical seam carving
        carvable_image = CarvableImage(self.original_image)
        carvable_image.energy_function = EnergyCalculator.squared_diff
        carvable_image.seam_function = SeamFinder.find_seam
        # carved_data = carvable_image.seam_carve(num_seams).img.mat
        carved_data = carvable_image.interactive_seam_carve(num_v_seams).img.mat
        
        # self.carved_image = Image(carved_data)
        # self._display_image(self.carved_image.mat, self.carved_image_label)
        
        # Horizontal seam carving
        carvable_image_hor = CarvableImage(Image(carved_data))
        carvable_image_hor.img.mat = cv2.rotate(carvable_image_hor.img.mat, cv2.ROTATE_90_CLOCKWISE)
        carvable_image_hor.energy_function = EnergyCalculator.squared_diff
        carvable_image_hor.seam_function = SeamFinder.find_seam
        carved_data_hor = carvable_image_hor.interactive_seam_carve(num_h_seams).img.mat
        carved_data_hor = cv2.rotate(carved_data_hor, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        self.carved_image = Image(carved_data_hor)
        self._display_image(self.carved_image.mat, self.carved_image_label)
        
    
    
        
    def start_seam_carving(self):
        if not self.original_image:
            return

        try:
            num_seams = int(self.seams_input.text())
            if num_seams < 1:
                return
        except ValueError:
            return

        carvable_image = CarvableImage(self.original_image)
        carvable_image.energy_function = EnergyCalculator.squared_diff
        carvable_image.seam_function = SeamFinder.find_seam
        carved_data = carvable_image.seam_carve(num_seams).img.mat

        self.carved_image = Image(carved_data)
        self._display_image(self.carved_image.mat, self.carved_image_label)

    def _display_image(self, image_data, label):
        height, width, channel = image_data.shape
        bytes_per_line = 3 * width
        q_img = QImage(image_data.data, width, height, bytes_per_line, QImage.Format_BGR888)
        label.setPixmap(QPixmap.fromImage(q_img))


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
