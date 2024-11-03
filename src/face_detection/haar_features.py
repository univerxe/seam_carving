class HaarFeature:
    def __init__(self, feature_type, position, width, height):
        self.feature_type = feature_type
        self.position = position  # Relative positions
        self.width = width
        self.height = height

    def compute_feature(self, integral_img, top_left, window_size):
        x, y = self.position
        w, h = self.width, self.height
        x = int(top_left[0] + x * window_size[0])
        y = int(top_left[1] + y * window_size[1])
        w = int(w * window_size[0])
        h = int(h * window_size[1])

        if self.feature_type == "two_horizontal":
            mid_w = w // 2
            A = self._sum_region(integral_img, x, y, mid_w, h)
            B = self._sum_region(integral_img, x + mid_w, y, mid_w, h)
            return A - B

        if self.feature_type == "two_vertical":
            mid_h = h // 2
            A = self._sum_region(integral_img, x, y, w, mid_h)
            B = self._sum_region(integral_img, x, y + mid_h, w, mid_h)
            return A - B

        if self.feature_type == "three_horizontal":
            mid_w = w // 3
            A = self._sum_region(integral_img, x, y, mid_w, h)
            B = self._sum_region(integral_img, x + mid_w, y, mid_w, h)
            C = self._sum_region(integral_img, x + 2 * mid_w, y, mid_w, h)
            return A - B - C

        if self.feature_type == "three_vertical":
            mid_h = h // 3
            A = self._sum_region(integral_img, x, y, w, mid_h)
            B = self._sum_region(integral_img, x, y + mid_h, w, mid_h)
            C = self._sum_region(integral_img, x, y + 2 * mid_h, w, mid_h)
            return A - B - C

        raise ValueError(f"Invalid feature type: {self.feature_type}")

    def _sum_region(self, integral_img, x, y, w, h):
        # Coordinates in the integral image
        x1, y1 = x, y
        x2, y2 = x + w, y + h
        # Sum calculation using the integral image
        A = integral_img[y1, x1]
        B = integral_img[y1, x2]
        C = integral_img[y2, x1]
        D = integral_img[y2, x2]
        return D - B - C + A
