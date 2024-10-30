from copy import deepcopy

import numpy as np
import cv2
from typing import Optional

from tqdm import trange

from src.algorithms.energy import EnergyCalculator
from src.algorithms.seam import SeamFinder


class Image(object):
    def __init__(self, mat: np.ndarray):
        self._mat = deepcopy(mat)
        self._validate_mat()

    @property
    def mat(self) -> np.ndarray:
        return self._mat

    @mat.setter
    def mat(self, value: np.ndarray):
        self._mat = value
        self._validate_mat()

    def _validate_mat(self):
        if not isinstance(self.mat, np.ndarray):
            raise ValueError(
                f"Expected: `np.ndarray` for `mat`, but got: {type(self.mat)}"
            )

        if len(self.mat.shape) != 3:
            raise ValueError(
                f"Image must be of shape (H, W, 3), but got: {self.mat.shape}"
            )

        if self.mat.dtype != np.uint8:
            raise ValueError(f"Image must be of type uint8, but got: {self.mat.dtype}")

    def save(self, path: str):
        cv2.imwrite(path, self.mat)


class CarvableImage(object):
    """
    Container for an image to perform seam-based operations.
    """

    @classmethod
    def from_path(cls, path: str):
        try:
            mat = cv2.imread(path)
        except Exception as e:
            raise ValueError(f"Failed to read '{path}': {e}")

        return cls(mat)

    def __init__(self, mat: np.ndarray):
        self._image = Image(mat)
        self._energy: Optional[np.ndarray] = None

    @property
    def mat(self) -> np.ndarray:
        """
        Return the data of the image.
        """
        return self._image.mat

    @property
    def shape(self) -> tuple[int, int]:
        """
        Return the shape of the image.

        Returns:
            tuple[int, int]: The height and width of the image.
        """
        return self.mat.shape[0], self.mat.shape[1]

    @property
    def channels(self) -> int:
        """
        Return the number of channels in the image.

        Returns:
            int: The number of channels in the image.
        """
        return self.mat.shape[2]

    @property
    def energy(self) -> Optional[np.ndarray]:
        return self._energy

    @energy.setter
    def energy(self, value: np.ndarray):
        self._energy = value
        self._validate_energy()

    def _validate_energy(self):
        assert self.energy is not None
        assert self.energy.shape == self.shape, (
            f"Energy map must have the same shape as the image: "
            f"{self.energy.shape} != {self.shape}"
        )

    def _find_seam(self) -> np.ndarray:
        """
        Find the seam with the lowest energy in the image.

        Returns:
            np.ndarray: The seam with the lowest energy.
        """
        assert (
            self.energy is not None
        ), "Energy map must be calculated before finding a seam."
        return SeamFinder.find_naive(self.energy)

    def _remove_seam(self, seam: np.ndarray):
        """
        Remove the seam from the image.

        Args:
            seam (np.ndarray): The seam to remove.
        """
        h, w, c = self.mat.shape
        new_mat = np.zeros(
            (h, w - 1, c),
            dtype=np.uint8,
        )

        for i in range(h):
            j = seam[i]
            new_mat[i, :, 0] = np.delete(self.mat[i, :, 0], j)
            new_mat[i, :, 1] = np.delete(self.mat[i, :, 1], j)
            new_mat[i, :, 2] = np.delete(self.mat[i, :, 2], j)

        self._image.mat = new_mat

    def calculate_energy(self, method: str):
        energy_func = getattr(EnergyCalculator, method)
        self.energy = energy_func(self.mat)

    def carve(self, num_iter: int, energy_method: str = "laplacian"):
        """
        Carve the image by removing seams.

        Args:
            num_iter (int): The number of seams to remove.
            energy_method (str): The method to use to calculate the energy of the image.
        """

        for _ in trange(num_iter, ncols=80):
            if hasattr(EnergyCalculator, energy_method):
                self.calculate_energy(energy_method)
            seam = self._find_seam()
            self._remove_seam(seam)

    def show(self):
        """
        Display the image.
        """
        cv2.imshow("Carvable Image", self.mat)
