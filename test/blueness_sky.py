from skimage.io import imread
import numpy as np
from scipy import ndimage

def SkyDetection(image):

    image = imread(image)
    R = image[:,:,0]
    G = image[:, :, 1]
    B = image[:, :, 2]

    # Mask for Sky with blueness factor, Change parameters for your Image
    blueness = B - np.maximum(R,G)
    brightness = np.maximum(np.maximum(R,G),B)
    mask = (blueness > 30) #* (brightness > 140)
    labeled_array, num_features = ndimage.label(mask)
    id = labeled_array[0,R.shape[1]-1]

    # Sky Mask
    mask = np.zeros_like(mask)
    mask[labeled_array==id]=1

    return mask

SkyDetection("C:\Master/temp/k2014-12-23_1128_VKA_6222.JPG")