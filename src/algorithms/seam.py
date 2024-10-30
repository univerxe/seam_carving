import numba
import numpy as np


class SeamFinder(object):
    """
    Find the seam with the lowest energy in an image.
    """

    @staticmethod
    @numba.jit
    def find_naive(mat: np.ndarray) -> np.ndarray:
        """
        Find the seam with the lowest energy in an image using the naive method.

        Args:
            mat (np.ndarray): The image to find the seam in.

        Returns:
            np.ndarray: The seam with the lowest energy.
        """
        assert len(mat.shape) == 2, "The input energy map must be a 2D matrix."

        h, w = mat.shape
        cum_energy = np.zeros_like(mat)
        cum_energy[0] = mat[0]

        for i in range(1, h):
            for j in range(w):
                left = cum_energy[i - 1, j - 1] if j > 0 else np.inf
                middle = cum_energy[i - 1, j]
                right = cum_energy[i - 1, j + 1] if j < mat.shape[1] - 1 else np.inf

                cum_energy[i, j] = mat[i, j] + np.min(np.array([left, middle, right]))

        seam = np.zeros(h, dtype=np.int32)
        seam[-1] = np.argmin(cum_energy[-1])

        for i in range(h - 2, -1, -1):
            j = seam[i + 1]
            left = cum_energy[i, j - 1] if j > 0 else np.inf
            middle = cum_energy[i, j]
            right = cum_energy[i, j + 1] if j < mat.shape[1] - 1 else np.inf

            seam[i] = j + np.argmin(np.array([left, middle, right])) - 1

        return seam
