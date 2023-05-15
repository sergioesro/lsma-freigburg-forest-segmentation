import numpy as np
import cv2


class RGBHistogram:
	def __init__(self, bins):
		self.bins = bins

	def describe(self, image):
		hist = cv2.calcHist([image], [0, 1, 2],
			None, self.bins, [0, 256, 0, 256, 0, 256])
		hist = cv2.normalize(hist,hist)

		return hist.flatten()


class LabHistogram:
    def __init__(self, bins):
        # store the number of bins for the histogram
        self.bins = bins

    def describe(self, image, mask=None):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        hist = cv2.calcHist([lab], [0, 1, 2], mask, self.bins,
            [0, 256, 0, 256, 0, 256])

        hist = cv2.normalize(hist,hist).flatten()

        return hist

	
class Searcher:
	def __init__(self, index):
		# store our index of images
		self.index = index

	def search(self, queryFeatures):
		# initialize our dictionary of results
		results = {}

		# loop over the index
		for (k, features) in self.index.items():
			d = self.chi2_distance(features, queryFeatures)
			results[k] = d
		results = sorted([(v, k) for (k, v) in results.items()])
		return results

	def chi2_distance(self, histA, histB, eps = 1e-10):
		# compute the chi-squared distance
		d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
			for (a, b) in zip(histA, histB)])

		# return the chi-squared distance
		return d