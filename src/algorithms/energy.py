import cv2
import numba
import numpy as np


class EnergyCalculator(object):
    """
    Calculate the energy of an image.

    Args:
        np.ndarray: The image to calculate the energy of.

    Returns:
        np.ndarray: The energy of the image with the same shape as the input.
    """

    @staticmethod
    # @numba.jit
    def sobel(mat: np.ndarray) -> np.ndarray:
        """
        Calculate the energy of an image using the Sobel operator.

        Args:
            mat (np.ndarray): The image to calculate the energy of.

        Returns:
            np.ndarray: The energy of the image with the same shape as the input.
        """
        b, g, r = cv2.split(mat.astype(np.float32))

        # Calculate the gradient in the x and y directions.
        e_b = np.abs(cv2.Sobel(b, -1, 1, 0)) + np.abs(cv2.Sobel(b, -1, 0, 1))
        e_g = np.abs(cv2.Sobel(g, -1, 1, 0)) + np.abs(cv2.Sobel(g, -1, 0, 1))
        e_r = np.abs(cv2.Sobel(r, -1, 1, 0)) + np.abs(cv2.Sobel(r, -1, 0, 1))

        return np.divide(e_b + e_g + e_r, 3)

    @staticmethod
    # @numba.jit
    def scharr(mat: np.ndarray) -> np.ndarray:
        """
        Calculate the energy of an image using the Scharr operator.

        Args:
            mat (np.ndarray): The image to calculate the energy of.

        Returns:
            np.ndarray: The energy of the image with the same shape as the input.
        """
        b, g, r = cv2.split(mat.astype(np.float32))

        # Calculate the gradient in the x and y directions.
        e_b = np.abs(cv2.Scharr(b, -1, 1, 0)) + np.abs(cv2.Scharr(b, -1, 0, 1))
        e_g = np.abs(cv2.Scharr(g, -1, 1, 0)) + np.abs(cv2.Scharr(g, -1, 0, 1))
        e_r = np.abs(cv2.Scharr(r, -1, 1, 0)) + np.abs(cv2.Scharr(r, -1, 0, 1))

        return np.divide(e_b + e_g + e_r, 3)

    @staticmethod
    # @numba.jit
    def laplacian(mat: np.ndarray) -> np.ndarray:
        """
        Calculate the energy of an image using the Laplacian operator.

        Args:
            mat (np.ndarray): The image to calculate the energy of.

        Returns:
            np.ndarray: The energy of the image with the same shape as the input.
        """
        b, g, r = cv2.split(mat.astype(np.float32))

        # Calculate the gradient in the x and y directions.
        e_b = np.abs(cv2.Laplacian(b, -1))
        e_g = np.abs(cv2.Laplacian(g, -1))
        e_r = np.abs(cv2.Laplacian(r, -1))

        return np.divide(e_b + e_g + e_r, 3)

    @staticmethod
    # @numba.jit
    def squared_diff(mat: np.ndarray) -> np.ndarray:
        """
        Calculate the energy of an image using the squared difference.

        Args:
            mat (np.ndarray): The image to calculate the energy of.

        Returns:
            np.ndarray: The energy of the image with the same shape as the input.
        """
        mat = mat.astype(np.float32)
        b = mat[:, :, 0]
        g = mat[:, :, 1]
        r = mat[:, :, 2]

        # Calculate the squared difference between the pixels.
        e_b = np.abs(b - np.roll(b, 1, axis=0)) + np.abs(b - np.roll(b, 1, axis=1))
        e_g = np.abs(g - np.roll(g, 1, axis=0)) + np.abs(g - np.roll(g, 1, axis=1))
        e_r = np.abs(r - np.roll(r, 1, axis=0)) + np.abs(r - np.roll(r, 1, axis=1))

        return np.divide(e_b + e_g + e_r, 3)
