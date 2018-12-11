import cv2
import numpy as np
from panorama import Stitcher
import imutils

# képek beolvasása
kep1 = cv2.imread("1.jpg")
kep2 = cv2.imread("2.jpg")
#átméretezés
kep1=imutils.resize(kep1, height=400)
kep2=imutils.resize(kep2, height=400)
#összeillesztés
stitcher = Stitcher()
(panorama, kpontok) = stitcher.stitch([kep1, kep2], showMatches=True)

panorama = imutils.resize (panorama, height = 400)
# megjelenítés és mentés
cv2.imshow("első kép", kep1)
cv2.imshow("második kép", kep2)
cv2.imshow("Kulcspontok", kpontok)
cv2.imshow("Panoramakep", panorama)
cv2.imwrite ("Panoramakep.png", panorama)
cv2.waitKey(0)