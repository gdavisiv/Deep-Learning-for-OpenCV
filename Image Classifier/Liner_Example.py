#Import neccessary packages for Liner_Example.py
import numpy as np
import cv2

#initialize  the class labels and set the seed of pseudorandom
#number generator to reproduce the results
lables= ["dog", "cat", "panda"]
np.random.seed(1)

#randomly initialize our weight matrix and bias vectors -- in a *real* world training and classification task
W = np.random.randn(3, 3702)
b = np.random.randn(3)

