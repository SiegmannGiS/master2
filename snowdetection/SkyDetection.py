from PIL import Image, ImageFilter
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

def OpenImage(image):
    # Open Reference Image
    img = Image.open(image)

    if image[-4:] == ".jpg":
        R, G, B = img.split()
        R = np.asarray(R, dtype=float)
        G = np.asarray(G, dtype=float)
        B = np.asarray(B, dtype=float)
    elif image[-4:] == ".png":
        R, G, B, A = img.split()
        R = np.asarray(R, dtype=float)
        G = np.asarray(G, dtype=float)
        B = np.asarray(B, dtype=float)

    return R,G,B

def SkyDetection(image):

    R, G, B = OpenImage(image)

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

def CloudCover(BlueSkyMask, image):

    R, G, B = OpenImage(image)
    blueness = B - np.maximum(R,G)

    BlueSkyPixel = (np.sum(np.sum(BlueSkyMask))).astype(float)
    ImagePixel = np.zeros_like(B)
    ImagePixel[blueness > 30] = 1
    ImageSkyPixel = ImagePixel*BlueSkyMask
    ImageSkyPixel = (np.sum(np.sum(ImageSkyPixel))).astype(float)
    CloudCoverPercent = 100-((ImageSkyPixel/BlueSkyPixel)*100.)
    print "[+] Cloud Cover:", CloudCoverPercent,"%"
    return CloudCoverPercent