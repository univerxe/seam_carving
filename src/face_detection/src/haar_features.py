from typing import Tuple
import numpy as np
from numpy.typing import NDArray

class HaarFeature:
    def __init__(self, feature_type: str, position: Tuple[int, int], width: int, height: int) -> None:
        """
        Initialize a Haar-like feature.

        Parameters:
        feature_type (str): The type of Haar-like feature ("two_horizontal", "two_vertical", "three_horizontal", "three_vertical").
        position (Tuple[int, int]): The position of the feature relative to the window as a percentage of the window size.
        width (int): The width of the feature as a percentage of the window size.
        height (int): The height of the feature as a percentage of the window size.
        """
        self.feature_type = feature_type
        self.position = position
        self.width = width
        self.height = height

    def compute_feature(self, integral_img: NDArray[np.float16], top_left: Tuple[int, int], window_size: Tuple[int, int]) -> float:
        """
        Compute the feature's value using an integral image.

        Parameters:
        integral_img (NDArray[np.float16]): The integral image.
        top_left (Tuple[int, int]): The top-left corner of the detection window.
        window_size (Tuple[int, int]): The size of the detection window.

        Returns:
        float: The computed feature value.
        """
        x, y = self.position
        w, h = self.width, self.height
        x = int(top_left[0] + x * window_size[0])
        y = int(top_left[1] + y * window_size[1])
        w = int(w * window_size[0])
        h = int(h * window_size[1])

        if self.feature_type == "two_horizontal":
            return self._compute_two_horizontal(integral_img, x, y, w, h)
        elif self.feature_type == "two_vertical":
            return self._compute_two_vertical(integral_img, x, y, w, h)
        elif self.feature_type == "three_horizontal":
            return self._compute_three_horizontal(integral_img, x, y, w, h)
        elif self.feature_type == "three_vertical":
            return self._compute_three_vertical(integral_img, x, y, w, h)
        else:
            raise ValueError(f"Invalid feature type: {self.feature_type}")

    def _compute_two_horizontal(self, integral_img: NDArray[np.float16], x: int, y: int, w: int, h: int) -> float:
        mid_w = w // 2
        A = self._sum_region(integral_img, x, y, mid_w, h)
        B = self._sum_region(integral_img, x + mid_w, y, mid_w, h)
        return A - B

    def _compute_two_vertical(self, integral_img: NDArray[np.float16], x: int, y: int, w: int, h: int) -> float:
        mid_h = h // 2
        A = self._sum_region(integral_img, x, y, w, mid_h)
        B = self._sum_region(integral_img, x, y + mid_h, w, mid_h)
        return A - B

    def _compute_three_horizontal(self, integral_img: NDArray[np.float16], x: int, y: int, w: int, h: int) -> float:
        mid_w = w // 3
        A = self._sum_region(integral_img, x, y, mid_w, h)
        B = self._sum_region(integral_img, x + mid_w, y, mid_w, h)
        C = self._sum_region(integral_img, x + 2 * mid_w, y, mid_w, h)
        return A - B + C

    def _compute_three_vertical(self, integral_img: NDArray[np.float16], x: int, y: int, w: int, h: int) -> float:
        mid_h = h // 3
        A = self._sum_region(integral_img, x, y, w, mid_h)
        B = self._sum_region(integral_img, x, y + mid_h, w, mid_h)
        C = self._sum_region(integral_img, x, y + 2 * mid_h, w, mid_h)
        return A - B + C

    def _sum_region(self, integral_img: NDArray[np.float16], x: int, y: int, w: int, h: int) -> float:
        """
        Calculate the sum of the pixel values within the rectangle specified.
        Utilizes the integral image to perform this calculation efficiently.

        Parameters:
        integral_img (NDArray[np.float16]): The integral image.
        x, y (int, int): The top-left corner of the rectangle.
        w, h (int, int): Width and height of the rectangle.

        Returns:
        float: The sum of the pixel values within the rectangle.
        """
        A = integral_img[y, x]
        B = integral_img[y, x + w]
        C = integral_img[y + h, x]
        D = integral_img[y + h, x + w]
        return D - B - C + A
