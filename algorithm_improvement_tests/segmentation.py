import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from skimage.color import rgb2hsv
from skimage.io import imread
from skimage.morphology import watershed, disk
from skimage import data
from skimage.filters import rank
from skimage.util import img_as_ubyte
from scipy import ndimage as ndi
import scipy as sp

def percentage(part, whole):
    return 100 * float(part)/float(whole)


img = imread("C:\Master\images/vernagtferner14-16/2015-10-12_08-30_0.jpg")

hsv = rgb2hsv(img)

value = img_as_ubyte(hsv[:, :, 2])

denoised = sp.misc.imresize(value, 0.10) / 255.

# find continuous region (low gradient -
# where less than 10 for this image) --> markers
# disk(5) is used here to get a more smooth image
markers = rank.gradient(denoised, disk(5)) < 50
markers = ndi.label(markers)[0]

# local gradient (disk(2) is used to keep edges thin)
gradient = rank.gradient(denoised, disk(2))

# process the watershed
labels = watershed(gradient, markers)

print "done"