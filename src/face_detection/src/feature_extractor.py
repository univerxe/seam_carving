import logging
import numpy as np
from typing import List, Any

from src.integral_image import compute_integral_image
from src.utils import sliding_window

def extract_features(images: List[np.ndarray], feature_list: List[Any]) -> np.ndarray:
    """
    Extract features from a list of images using a provided list of feature descriptor objects.

    Parameters:
    images (List[np.ndarray]): List of images to process.
    feature_list (List[Any]): List of feature descriptor objects that can compute features using an integral image.

    Returns:
    np.ndarray: Array of extracted feature vectors from all images.
    """
    feature_vectors = []
    for img_index, img in enumerate(images):
        try:
            integral_img = compute_integral_image(img)
            window_sizes = [(96, 96)]  # Example: Changed to a single window size for simplification
            step_size = 96

            for window_size in window_sizes:
                for (x, y, window) in sliding_window(img, step_size, window_size):
                    features = [feature.compute_feature(integral_img, (x, y), window_size) for feature in feature_list]
                    feature_vectors.append(features)

            logging.info(f"Extracted {len(feature_vectors)} features from image {img_index}")

        except Exception as e:
            logging.error(f"Error processing image {img_index}: {e}")

    return np.array(feature_vectors)
