import sys
import os
import cv2
import numpy as np
from PySide6.QtWidgets import (
    QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QLabel, 
    QFileDialog, QLineEdit, QApplication, QGroupBox
)
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt
from src.lib import Image, CarvableImage
from src.algorithms.energy import EnergyCalculator
from src.algorithms.seam import SeamFinder

class MainWindow(QMainWindow):
    """
    Main window class for interactive seam carving with side-by-side image display.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Seam Carving - Before and After Comparison")
        self.setGeometry(200, 200, 1000, 600)  # Set the window size
        self.original_image = None  # Original Image instance
        self.carved_image = None  # Carved Image instance
        
        # Central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        # Image controls group with centered buttons
        controls_group = QGroupBox("Image Controls")
        controls_layout = QVBoxLayout()
        controls_group.setLayout(controls_layout)
        main_layout.addWidget(controls_group, alignment=Qt.AlignCenter)
        
        # Center-align layout for buttons and input field
        button_layout = QHBoxLayout()
        button_layout.setAlignment(Qt.AlignCenter)

        # Load Image Button
        self.load_button = QPushButton("Load Image")
        self.load_button.setFixedWidth(150)
        self.load_button.clicked.connect(self.load_image)
        button_layout.addWidget(self.load_button)
        
        # Input for Number of Seams
        self.seams_input = QLineEdit()
        self.seams_input.setPlaceholderText("Enter number of seams to carve")
        self.seams_input.setFixedWidth(150)
        button_layout.addWidget(self.seams_input)
        
        # Interactive Seam Carving Button
        self.carve_button = QPushButton("Start Seam Carving")
        self.carve_button.setFixedWidth(150)
        self.carve_button.clicked.connect(self.start_seam_carving)
        button_layout.addWidget(self.carve_button)

        # Add button layout to controls layout
        controls_layout.addLayout(button_layout)

        # Horizontal layout for side-by-side image display
        image_display_group = QGroupBox("Image Comparison")
        image_layout = QHBoxLayout()
        image_display_group.setLayout(image_layout)
        main_layout.addWidget(image_display_group)

        # Original Image Display Label with "Before" text
        original_layout = QVBoxLayout()
        original_text = QLabel("Before (Original Image)")
        original_text.setAlignment(Qt.AlignCenter)
        self.original_image_label = QLabel()
        self.original_image_label.setAlignment(Qt.AlignCenter)
        self.original_image_label.setMinimumSize(700, 700)
        self.original_image_label.setStyleSheet("border: 2px solid gray;")  # Add a border
        original_layout.addWidget(original_text)
        original_layout.addWidget(self.original_image_label)
        image_layout.addLayout(original_layout)

        # Carved Image Display Label with "After" text
        carved_layout = QVBoxLayout()
        carved_text = QLabel("After (Carved Image)")
        carved_text.setAlignment(Qt.AlignCenter)
        self.carved_image_label = QLabel()
        self.carved_image_label.setAlignment(Qt.AlignCenter)
        self.carved_image_label.setMinimumSize(700, 700)
        self.carved_image_label.setStyleSheet("border: 2px solid gray;")  # Add a border
        carved_layout.addWidget(carved_text)
        carved_layout.addWidget(self.carved_image_label)
        image_layout.addLayout(carved_layout)

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.xpm *.jpg *.bmp)")
        if file_path:
            self.original_image = Image.from_path(file_path)  # Load the image using the Image class
            self.display_image(self.original_image.mat, self.original_image_label)

    def start_seam_carving(self):
        if self.original_image is None:
            print("No image loaded.")
            return

        # Get the number of seams from user input
        try:
            num_seams = int(self.seams_input.text())
            if num_seams < 1:
                print("Please enter a positive integer for seams.")
                return
        except ValueError:
            print("Please enter a valid integer for seams.")
            return

        # Create a CarvableImage instance and set energy and seam functions
        carvable_image = CarvableImage(self.original_image)
        carvable_image.energy_function = EnergyCalculator.squared_diff
        carvable_image.seam_function = SeamFinder.find_seam

        # Perform seam carving interactively
        carved_image = carvable_image.interactive_seam_carve(num_seams).img.mat
        self.carved_image = Image(carved_image)  # Update the carved image
        self.display_image(self.carved_image.mat, self.carved_image_label)

    def display_image(self, image_data, label):
        # Convert the image data to QImage for displaying in QLabel
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
