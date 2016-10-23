from skimage import data
from skimage import transform as tf
from skimage.feature import (match_descriptors, corner_harris,
                             corner_peaks, ORB, plot_matches)
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import numpy as np


test = np.array([2,6,4,8,9])

descriptor_extractor = ORB(n_keypoints=200)

descriptor_extractor.detect_and_extract(test)