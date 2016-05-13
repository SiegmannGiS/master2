from skimage import feature
from skimage.io import imread
import matplotlib.pyplot as plt
import numpy as np
import os

def edge(pathfile):
    image = imread(pathfile, as_grey=True)
    image = image[190:381, :]
    edges = feature.canny(image, sigma=2)
    num = np.sum(edges)
    return num, edges

path = "C:/master/temp/"

ref = "C:/master/temp/k2016-03-26_1125_VKA_7414.JPG"
list = []


num, refrast = edge(ref)
list.append(num)


for file in os.listdir(path):
    if file[-4:] == ".JPG":
        pathfile = os.path.join(path,file)

        num, rast = edge(pathfile)
        list.append(num)
        prozent = float(num)/float(list[0])*100

        print file, num, prozent

        fig, axs = plt.subplots(2, 1)
        fig.suptitle("Comparison Canny Edge Detector, sigma=2")
        axs[0].imshow(refrast)
        axs[0].set_title("Reference Raster")
        axs[1].imshow(rast)
        axs[1].set_title("Compared Image")
        plt.tight_layout()
        plt.show()

