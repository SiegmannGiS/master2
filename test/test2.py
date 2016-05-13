from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from skimage.color import rgb2hsv
from skimage.io import imread

with np.load("C:\Master\settings\wallackhaus-nord/settings.npz") as settings:
    clearmask = settings["clearmask"]
    skymask = settings["skymask"]

img = imread("C:/master/temp/k2016-03-21_1655_VKA_7400.JPG")

hsv = rgb2hsv(img)

skychannel = (hsv[:, :, 1] * 100).astype(int)
clearchannel = (hsv[:, :, 2] * 100).astype(int)

print skychannel[skymask] < 30
print clearchannel[clearmask] < 50

plt.subplot(2, 2, 3)
plt.imshow(skychannel, vmin=0, vmax=100, cmap=cm.get_cmap("Greys"))
plt.title("Saturation")
plt.colorbar()

plt.subplot(2, 2, 4)
plt.imshow(clearchannel, vmin=0, vmax=100, cmap=cm.get_cmap("Greys"))
plt.title("Value")
plt.colorbar()

plt.subplot(2, 2, 1)
plt.imshow(img)
plt.colorbar()
plt.title("Image")

plt.subplot(2, 2, 2)
plt.imshow((hsv[:, :, 0] * 360).astype(int), vmin=0, vmax=360, cmap=cm.get_cmap("hsv"))
plt.title("Hue")
plt.colorbar()

plt.show()