import numpy as np
import imutils
import cv2

class Stitcher:
	def __init__(self):
		# OpenCv verzió ellenőrzés
		self.isv3 = imutils.is_cv3()

	def stitch(self, kepek, ratio=0.75, reprojThresh=4.0,
		showMatches=False):
		# képek megnyitása, kulcspontok kinyerése
		(kep2, kep1) = kepek
		(kpsA, featuresA) = self.JellemzoFelismer(kep1)
		(kpsB, featuresB) = self.JellemzoFelismer(kep2)

		# Közös pontok felismerése
		M = self.KulcspontIlleszt(kpsA, kpsB,
			featuresA, featuresB, ratio, reprojThresh)

		if M is None:
			return None

		# warping/transzformálás
		(matches, H, status) = M
		result = cv2.warpPerspective(kep1, H, (kep1.shape[1] + kep2.shape[1], kep1.shape[0]))
		result[0:kep2.shape[0], 0:kep2.shape[1]] = kep2

		# közös pontok megjelenítése
		if showMatches:
			vis = self.KiRajzol(kep1, kep2, kpsA, kpsB, matches,
				status)

			return (result, vis)

		return result
	    
	def JellemzoFelismer(self, image):
		# szürkeárnyalat
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		# OpenCv ellenőrzés
		if self.isv3:
			# jellemzők felismerése
			descriptor = cv2.xfeatures2d.SIFT_create()
			(kps, features) = descriptor.detectAndCompute(image, None)

		# OpenCv 2.4.X
		else:
			# kulcspont felismerés
			detector = cv2.FeatureDetector_create("SIFT")
			kps = detector.detect(gray)

			# jellemzők meghatározása
			extractor = cv2.DescriptorExtractor_create("SIFT")
			(kps, features) = extractor.compute(gray, kps)

		# NumPy vektorokba konvertálja
		kps = np.float32([kp.pt for kp in kps])

		return (kps, features)


	def KulcspontIlleszt(self, kpsA, kpsB, featuresA, featuresB,
		ratio, reprojThresh):
		# egyezések listája
		matcher = cv2.DescriptorMatcher_create("BruteForce")
		rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
		matches = []

		for m in rawMatches:
			# távolságok mérése és összehasonlítása
			if len(m) == 2 and m[0].distance < m[1].distance * ratio:
				matches.append((m[0].trainIdx, m[0].queryIdx))

		# computing a homography requires at least 4 matches
		if len(matches) > 4:
			# tényleges összeillesztés
			ptsA = np.float32([kpsA[i] for (_, i) in matches])
			ptsB = np.float32([kpsB[i] for (i, _) in matches])

			# homography/homográfia
			(H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
				reprojThresh)

			# visszatér az egyezésekkel, a homográfiamátrixszal és a pontok státuszával
			return (matches, H, status)

		return None

	def KiRajzol(self, kep1, kep2, kpsA, kpsB, matches, status):
		# kimeneti kép
		(hA, wA) = kep1.shape[:2]
		(hB, wB) = kep2.shape[:2]
		vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
		vis[0:hA, 0:wA] = kep1
		vis[0:hB, wA:] = kep2

		for ((trainIdx, queryIdx), s) in zip(matches, status):
			
			if s == 1:
				# kirajzolja az egyezéseket
				ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
				ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
				cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

		return vis