import os
import cv2
import numpy as np
from typing import List, Tuple
from numpy.typing import NDArray

def load_images_from_folder(
    folder: str,
    image_size: Tuple[int, int] | None,
    color_mode: int = cv2.IMREAD_GRAYSCALE
) -> NDArray[np.float16]:
    """
    Load images from a specified folder, optionally resizing them and adjusting the color mode.

    Parameters:
    folder (str): Path to the folder containing images.
    image_size (Optional[Tuple[int, int]]): Desired size of output images (width, height). If None, images are not resized.
    color_mode (int): Color mode in which to load images. Default is grayscale (cv2.IMREAD_GRAYSCALE).

    Returns:
    NDArray[np.float16]: Numpy array of images loaded and processed according to specified parameters.
    """
    images: List[NDArray[np.float16]] = []

    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        img = cv2.imread(filepath, color_mode)

        if img is not None:
            if image_size is not None:
                img = cv2.resize(img, image_size)  # Resize image if size specified

            images.append(img.astype(np.float16))  # Append the processed image as a float array

    return np.array(images)  # Return a single numpy array containing all images
