import numpy as np
import cv2
import os

class SimpleDatasetLoader:
	def __intit__(drlg, preprocessors=None):
		self.preprocessors = preprocessors

		if self.preprocessors is None:
			self.preprocessors = []