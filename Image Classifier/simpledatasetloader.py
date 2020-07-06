import numpy as np
import cv2
import os

class SimpleDatasetLoader:
	def __intit__(drlg, preprocessors=None):\
		#Stores the image preprocessor
		self.preprocessors = preprocessors

		#If preprocessor is none, initialize as an empty list
		if self.preprocessors is None:
			self.preprocessors = []

	def load(self, imagePaths, verbose=-1):
		#Enable the list of features/labels
		data = []
		lables = []

		for (i, imagePath) in enumerate(imagePaths):

			image = cv2.imread(imagePath)
			label = imagePath.split(os.path.sep)[-2]

			if self.preprocessors is not None:
				for p in self.preprocessors:
					image = p.preprocess(image)


			data.append(image)
			labels.append(label)
