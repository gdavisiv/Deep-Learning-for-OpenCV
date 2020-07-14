#import packages
#implementation of k-NN algo provided by scikit-learn
from sklearn.neighbors import KNeighborsClassifier
#util that converts labels as strings to integers
from sklearn.preprocessing import LabelEncoder
#helps create the training/testing splits
from sklearn.model_selection import train_test_split
#helps evaluate the performance of classifier and print out formatted table of results
from sklearn.metrics import classification_report
from pyimagesearch.preprocessing import SimplePreprocessor
from pyimagesearch.datasets import SimpleDatasetLoader
from imutils import paths
import argparse

