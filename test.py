import cv2

from src.algorithms.energy import EnergyCalculator
from src.algorithms.seam import SeamFinder
from src.lib import CarvableImage

carvable = CarvableImage.from_path("images/castle_small.png")

carvable.energy_function = EnergyCalculator.squared_diff
carvable.seam_function = SeamFinder.find_seam
num_seams = int(input("Enter the number of seams: "))

carved = carvable.seam_carve(num_seams, show_progress=True)

carvable.img.show()
carved.img.show(f"Seam carving x {num_seams} times")
cv2.waitKey(0)
