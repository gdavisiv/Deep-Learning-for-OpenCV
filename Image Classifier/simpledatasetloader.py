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

