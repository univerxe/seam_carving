from copy import deepcopy

import cv2

from src.lib import CarvableImage

img = CarvableImage.from_path("images/castle_small.png")
original = deepcopy(img.mat)
cv2.imshow("Original", original)

img.carve(128, "squared_diff")
carved = deepcopy(img.mat)
cv2.imshow("Squared Diff", carved)


cv2.waitKey(0)
