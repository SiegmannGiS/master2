from __future__ import print_function

import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import rgb2hsv

from skimage.segmentation import felzenszwalb, slic, quickshift
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float




img = imread("C:\Master\images/vernagtferner14-16/2015-10-12_08-30_0.jpg")
hsv = rgb2hsv(img)
img = img_as_float(hsv[:, :, 2])
#img = img_as_float(img[::5,::5])

segments_fz = felzenszwalb(img, scale=500, sigma=0.5, min_size=50)

print("done")

print("hallo")
