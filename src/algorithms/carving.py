import numba
import numpy as np


@numba.njit
def carve_seam(mat: np.ndarray, seam: np.ndarray) -> np.ndarray:
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

def carve_seam_enlarge(mat: np.ndarray, seam: np.ndarray) -> np.ndarray:
    """
    Add a seam from an image.

    Args:
        mat (np.ndarray): The image to remove the seam from.
        seam (np.ndarray): The seam to remove.

    Returns:
        np.ndarray: The image with the seam added.
    """
    assert len(mat.shape) == 3, "The input image must be a 3D matrix."

    h, w, c = mat.shape
    assert len(seam) == h, "The seam must have the same height as the image."

    enlarged = np.zeros((h, w + 1, c), dtype=np.uint8)

    for y in range(h):
        x = seam[y]
        enlarged[y, :x] = mat[y, :x]

        if x < w - 1:
            avg_pixel = (mat[y, x].astype(np.int32) + mat[y, x + 1].astype(np.int32)) // 2
            enlarged[y, x] = avg_pixel.astype(np.uint8)
            enlarged[y, x + 1] = mat[y, x]
            enlarged[y, x + 2:] = mat[y, x + 1:]
        else:
            enlarged[y, x] = mat[y, x]
            enlarged[y, x + 1] = mat[y, x]

    return enlarged.astype(np.uint8)