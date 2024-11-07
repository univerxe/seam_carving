import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Generator

def normalize_images(images: NDArray[np.float16]) -> NDArray[np.float16]:
    """
    Normalize a batch of images by scaling their pixel values to the range [0, 1].

    Parameters:
    images (NDArray[np.float16]): Numpy array of images.

    Returns:
    NDArray[np.float16]: Normalized numpy array of images.
    """
    return images / 255.0

def sliding_window(
    image: NDArray[np.float16], 
    step_size: int, 
    window_size: Tuple[int, int]
) -> Generator[Tuple[int, int, NDArray[np.float16]], None, None]:
    """
    Generate a sliding window across the image with a specified step size and window size.

    Parameters:
    image (NDArray[np.float16]): The image to slide the window across.
    step_size (int): The number of pixels to move the window each step along both x and y axes.
    window_size (Tuple[int, int]): The dimensions of the window (width, height).

    Yields:
    Tuple[int, int, NDArray[np.float16]]: The top-left x and y coordinates of the window and the window itself.
    """
    for y in range(0, image.shape[0] - window_size[1] + 1, step_size):
        for x in range(0, image.shape[1] - window_size[0] + 1, step_size):
            yield (x, y, image[y: y + window_size[1], x: x + window_size[0]])

