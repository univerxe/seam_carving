from copy import deepcopy

import cv2
from tqdm import trange

from src.algorithms.energy import EnergyCalculator
from src.algorithms.seam import find_seam
from src.algorithms.carving import carving_seam

img = cv2.imread("images/castle_small.png")
carved = img.copy()

energy_map = EnergyCalculator.squared_diff(img)
cv2.imshow("Energy Map", energy_map.astype("uint8"))
cv2.waitKey(0)

seam = find_seam(energy_map)
seam_img = img.copy()
for y, x in enumerate(seam):
    seam_img[y, x] = [0, 0, 255]
cv2.imshow("Seam", seam_img)
cv2.waitKey(0)

i: int = 0
for i in trange(100, ncols=100):
    energy_map = EnergyCalculator.squared_diff(carved)
    seam = find_seam(energy_map)
    carved = carving_seam(carved, seam)

cv2.imshow("Original", img)
cv2.imshow(f"Carved {i+1} times", carved)

cv2.waitKey(0)
