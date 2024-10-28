from copy import deepcopy

import numpy as np
import cv2


class Image(object):
    def __init__(self, mat: np.ndarray):
        self._mat = deepcopy(mat)
        self._validate_mat()

    @property
    def mat(self) -> np.ndarray:
        return self._mat

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
