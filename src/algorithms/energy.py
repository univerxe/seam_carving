import cv2
import numba
import numpy as np


class EnergyCalculator(object):

    @staticmethod
    @numba.njit
    def squared_diff(mat: np.ndarray) -> np.ndarray:
        assert len(mat.shape) == 3, "The input image must be a 3D matrix."
        w, h, _ = mat.shape

        intensity = np.zeros((w, h), dtype=np.float32)
        for y in range(mat.shape[0]):
            for x in range(mat.shape[1]):
                b, g, r = mat[y, x]
                intensity[y, x] = 0.299 * r + 0.587 * g + 0.114 * b

        energy_map = np.zeros_like(intensity, dtype=np.float32)

        # Handle the borders
        energy_map[0, :] = intensity[1, :] / 2.0
        energy_map[-1, :] = np.abs(intensity[-1, :] - intensity[-2, :])

        energy_map[:, 0] = intensity[:, 1] / 2.0
        energy_map[:, -1] = np.abs(intensity[:, -1] - intensity[:, -2])

        for y in range(1, w - 1):
            for x in range(1, h - 1):
                dy = (intensity[y + 1, x] - intensity[y - 1, x]) / 2.0
                dx = (intensity[y, x + 1] - intensity[y, x - 1]) / 2.0

                # Approximate by the sum of the absolute differences
                energy_map[y, x] = np.abs(dy) + np.abs(dx)

        return energy_map.astype(np.float32)


class _EnergyCalculator(object):
    """
    Calculate the energy of an image.

    Args:
        np.ndarray: The image to calculate the energy of.

    Returns:
        np.ndarray: The energy of the image with the same shape as the input.
    """

    @staticmethod
    @numba.jit(nopython=False, forceobj=True)
    def squared_diff_1c(mat: np.ndarray) -> np.ndarray:
        """
        Calculate the energy of an image using the squared difference.

        Args:
            mat (np.ndarray): The image to calculate the energy of.

        Returns:
            np.ndarray: The energy of the image with the same shape as the input.
        """
        mat = cv2.cvtColor(mat, cv2.COLOR_BGR2GRAY)
        w, h = mat.shape

        grad_y = np.zeros_like(mat, dtype=np.float32)
        grad_x = np.zeros_like(mat, dtype=np.float32)

        # Handle the borders
        grad_y[0, :] = mat[0, :]
        grad_y[-1, :] = mat[-2, :] - mat[-1, :]

        grad_x[:, 0] = mat[:, 0]
        grad_x[:, -1] = mat[:, -2] - mat[:, -1]

        for y in range(1, w - 1):
            for x in range(1, h - 1):
                grad_y[y, x] = (mat[y + 1, x] - mat[y - 1, x]) / 2
                grad_x[y, x] = (mat[y, x + 1] - mat[y, x - 1]) / 2

        return np.abs(grad_y) + np.abs(grad_x)

    @staticmethod
    @numba.njit
    def squared_diff(mat: np.ndarray) -> np.ndarray:
        """
        Calculate the energy of an image using the squared difference.

        Args:
            mat (np.ndarray): The image to calculate the energy of.

        Returns:
            np.ndarray: The energy of the image with the same shape as the input.
        """
        b, g, r = mat[:, :, 0], mat[:, :, 1], mat[:, :, 2]

        # Calculate the squared difference between the pixels.
        # Note: approximating by the sum of the absolute differences.
        e_b = np.zeros_like(b, dtype=np.uint8)
        e_g = np.zeros_like(g, dtype=np.uint8)
        e_r = np.zeros_like(r, dtype=np.uint8)

        for y in range(mat.shape[0]):
            for x in range(mat.shape[1]):
                # Current pixel values
                b_current, g_current, r_current = b[y, x], g[y, x], r[y, x]

                e_b[y, x] += abs(b_current - b[y - 1, x]) + abs(b_current - b[y, x - 1])
                e_g[y, x] += abs(g_current - g[y - 1, x]) + abs(g_current - g[y, x - 1])
                e_r[y, x] += abs(r_current - r[y - 1, x]) + abs(r_current - r[y, x - 1])

        return np.divide(e_b + e_g + e_r, 3).astype(np.float32)

    @staticmethod
    @numba.jit(nopython=False, forceobj=True)
    def sobel(mat: np.ndarray) -> np.ndarray:
        """
        Calculate the energy of an image using the Sobel operator.

        Args:
            mat (np.ndarray): The image to calculate the energy of.

        Returns:
            np.ndarray: The energy of the image with the same shape as the input.
        """
        b, g, r = mat[:, :, 0], mat[:, :, 1], mat[:, :, 2]

        # Calculate the Sobel operator of the image
        e_b = np.abs(cv2.Sobel(b, -1, 1, 0)) + np.abs(cv2.Sobel(b, -1, 0, 1))
        e_g = np.abs(cv2.Sobel(g, -1, 1, 0)) + np.abs(cv2.Sobel(g, -1, 0, 1))
        e_r = np.abs(cv2.Sobel(r, -1, 1, 0)) + np.abs(cv2.Sobel(r, -1, 0, 1))

        return np.divide(e_b + e_g + e_r, 3).astype(np.float32)

    @staticmethod
    @numba.jit(nopython=False, forceobj=True)
    def scharr(mat: np.ndarray) -> np.ndarray:
        """
        Calculate the energy of an image using the Scharr operator.

        Args:
            mat (np.ndarray): The image to calculate the energy of.

        Returns:
            np.ndarray: The energy of the image with the same shape as the input.
        """
        b, g, r = mat[:, :, 0], mat[:, :, 1], mat[:, :, 2]
        b_energy = np.abs(cv2.Scharr(b, -1, 1, 0)) + np.abs(cv2.Scharr(b, -1, 0, 1))
        g_energy = np.abs(cv2.Scharr(g, -1, 1, 0)) + np.abs(cv2.Scharr(g, -1, 0, 1))
        r_energy = np.abs(cv2.Scharr(r, -1, 1, 0)) + np.abs(cv2.Scharr(r, -1, 0, 1))

        return np.divide(b_energy + g_energy + r_energy, 3).astype(np.float32)

    @staticmethod
    @numba.jit(nopython=False, forceobj=True)
    def laplacian(mat: np.ndarray) -> np.ndarray:
        """
        Calculate the energy of an image using the Laplacian operator.

        Args:
            mat (np.ndarray): The image to calculate the energy of.

        Returns:
            np.ndarray: The energy of the image with the same shape as the input.
        """
        b, g, r = mat[:, :, 0], mat[:, :, 1], mat[:, :, 2]

        # Calculate the Laplacian of the image
        e_b = np.abs(cv2.Laplacian(b, -1))
        e_g = np.abs(cv2.Laplacian(g, -1))
        e_r = np.abs(cv2.Laplacian(r, -1))

        return np.divide(e_b + e_g + e_r, 3).astype(np.float32)
