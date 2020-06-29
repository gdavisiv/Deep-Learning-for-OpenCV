#import the needed libraries
import cv2

class SimplePreprocessor:
	def __intit__(self, width, height, inter=cv2.INTER_AREA):
	#stores the target image width, height, interpolation

	self.width = width
	self.height = height
	self.inter = inter

	def preprocess(self, image):
		#resize the image for a fixed resolution, ignore aspect ratio
		return cv2.resize(images, (self.width, self.height), interpolation=self.inter)