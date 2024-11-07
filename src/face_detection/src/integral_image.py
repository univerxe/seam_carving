import numpy as np
from numpy.typing import NDArray

def compute_integral_image(img: NDArray) -> NDArray:
    """
    Compute the integral image for a given 2D array.

    Parameters:
    img (ndarray): A 2D array representing the image.

    Returns:
    ndarray: A 2D array representing the integral image with an added row and column.
    """
    # Initialize an array with an extra row and column, using data type np.float64
    integral_img = np.zeros((img.shape[0] + 1, img.shape[1] + 1), dtype=np.float64)

    # Compute the cumulative sum along the rows and then along the columns
    integral_img[1:, 1:] = np.cumsum(np.cumsum(img, axis=0), axis=1)

    return integral_img
