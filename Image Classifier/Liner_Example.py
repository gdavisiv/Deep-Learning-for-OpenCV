#Import neccessary packages for Liner_Example.py
import numpy as np
import cv2

#initialize  the class labels and set the seed of pseudorandom
#number generator to reproduce the results
lables= ["dog", "cat", "panda"]
np.random.seed(1)

#randomly initialize our weight matrix and bias vectors -- in a *real* world training and classification task
W = np.random.randn(3, 3072)
b = np.random.randn(3)

#Load the example image/resize/flatten for "Feature Vector" representation
orig = cv2.imread("beagle.png")
image = cv2.resize(orig, (32, 32)).flatten()

#Computes the output scores via:
#Taking dot product between the weight matrix W and the input image pixel intensifies
scores = W.dot(image) + b

#Loops over the scores + labels and prints it to display
for(label, score) in zip(labels, scores):
	print("[INFO] {}: {:.2f}".format(label, score))

#draws the label(dog. cat, panda) with the highest scores on the image as our prediction
cv2.putText(orig, "Label: {}".format(labels[np.argmax(scores)]), 
	(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

cv2.imshow("Image", orig)
cv2.waitKey(0)
