#packages needed
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
import argparse

def sigmoid_activation(x):
	return 1.0 / (1 + np.esp(-x))

def sigmoid_deriv(x):
	return x * (1-x)