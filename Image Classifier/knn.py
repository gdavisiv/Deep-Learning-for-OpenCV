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

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-k", "--neighbors", type=int, default=1, help="# of nearest neighbors for classification")
ap.add_argument("-j", "--jobs", type=int, default=-1, help="# of jobs for k-NN distance (-1 uses all available cores)")
args = vars(ap.parse_args())

#get the list of images that will be distributed
print("[INFO] loading images...")
imagePaths = lists(paths.list_images(args["dataset"]))

#initialize the image preprocessor, load the dataset, reshape the data matrix
sp = SimplePreprocessor(32, 32)
sdl = SimpleDatasetLoader(preprocessors = [sp])
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.reshape((data.shape[0], 3072))

#show info on memory consumption of images
print("[INFO] fratures matric: {:..1f}MB".format(data.nbytes / (1024 * 1024.0)))