#import the neccessary packages
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

		#Loop over the input images
		for (i, imagePath) in enumerate(imagePaths):

			#load the image and extract the class label assuming, that
			#the path has the following format: 
			#/path/to/dataset/{class}/{image}.jpg
			image = cv2.imread(imagePath)
			label = imagePath.split(os.path.sep)[-2]

			#checks to see if preprocessors are not None,
			#if the check passes we loop over each preprocessor and squentially apply
			#them to the image
			if self.preprocessors is not None:
				for p in self.preprocessors:
					#This allows us to create a chain of preprocessors that can be applied to 
					#every image in the dataset
					image = p.preprocess(image)

			#Once the image is preprocessed we update the data/label lists
			data.append(image)
			labels.append(label)

			#provides a printed updated for every verbose image
			if verbose > 0 and i > 0 and (i+1) % verbose == 0:
				print("[INFO] processed {}/{}".format(i + 1, len(immagePaths)))

		#returns a tuple: single row of a table, of data/labels
		return (np.array(data), np.array(labels))