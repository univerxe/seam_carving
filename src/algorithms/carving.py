import numba
import numpy as np


@numba.njit
def carving_seam(mat: np.ndarray, seam: np.ndarray) -> np.ndarray:
    """
    Remove a seam from an image.

    Args:
        mat (np.ndarray): The image to remove the seam from.
        seam (np.ndarray): The seam to remove.

    Returns:
        np.ndarray: The image with the seam removed.
    """
    assert len(mat.shape) == 3, "The input image must be a 3D matrix."

    h, w, c = mat.shape
    assert len(seam) == h, "The seam must have the same height as the image."

    carved = np.zeros((h, w - 1, c), dtype=np.uint8)

    for y in range(h):
        x = seam[y]
        carved[y, :x] = mat[y, 0:x]
        carved[y, x:] = mat[y, x + 1 :]

    return carved.astype(np.uint8)
