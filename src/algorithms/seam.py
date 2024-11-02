import numba
import numpy as np


class SeamFinder(object):

    @staticmethod
    @numba.njit
    def find_seam(energy_map: np.ndarray) -> np.ndarray:
        """
        Find the seam with the lowest energy in an image.

        Args:
            energy_map (np.ndarray): The energy map of the image of shape (h, w).

        Returns:
            np.ndarray: The seam with the lowest energy of shape (h,).
        """
        assert len(energy_map.shape) == 2, "The input energy map must be a 2D matrix."

        h, w = energy_map.shape

        # Build the cumulative energy map
        cumulative_energy_map = np.zeros_like(energy_map)
        cumulative_energy_map[0] = energy_map[0]  # Initial value
        preceding = np.zeros(3)
        for y in range(1, h):
            for x in range(w):
                self = energy_map[y, x]

                # Previous row (cumulated)
                left = cumulative_energy_map[y - 1, x - 1] if x - 1 >= 0 else np.inf
                middle = cumulative_energy_map[y - 1, x]
                right = cumulative_energy_map[y - 1, x + 1] if x + 1 < w else np.inf

                preceding[0] = left
                preceding[1] = middle
                preceding[2] = right
                # The energy of the current pixel is the sum of its own energy and the minimum of the
                # three possible paths from the previous row to the current pixel.
                cumulative_energy_map[y, x] = self + np.min(preceding)

        # Find the seam with the lowest energy
        seam = np.zeros(h, dtype=np.int32)
        # The last pixel of the seam is the one with the lowest energy in the last row
        seam[-1] = np.argmin(cumulative_energy_map[-1])

        # y \in [h - 2, 0]
        for y in range(h - 2, -1, -1):  # From the second last row to the first row
            x = seam[y + 1]
            left = cumulative_energy_map[y, x - 1] if x - 1 >= 0 else np.inf
            middle = cumulative_energy_map[y, x]
            right = cumulative_energy_map[y, x + 1] if x + 1 < w else np.inf

            preceding[0] = left
            preceding[1] = middle
            preceding[2] = right
            seam[y] = x + np.argmin(preceding) - 1
            # assert 0 <= seam[y] < w, f"The seam must be within the image boundaries. ({y=})"

        return seam.astype(np.int32)
