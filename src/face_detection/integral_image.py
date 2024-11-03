import numpy as np


def compute_integral_image(img):
    # Initialize an array with extra row and column
    integral_img = np.zeros((img.shape[0] + 1, img.shape[1] + 1), dtype=np.float64)
    # Compute the cumulative sum
    integral_img[1:, 1:] = np.cumsum(np.cumsum(img, axis=0), axis=1)
    return integral_img
