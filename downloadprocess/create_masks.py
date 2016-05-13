import matplotlib.pyplot as plt
from skimage.io import imread
from PIL import Image
import numpy as np

name = "vernagtferner14-16"

sky = imread("C:/Master/settings/%s/sky.jpg" %name)
classification = imread("C:/Master/settings/%s/mask.jpg" %name)
image = imread("C:\Master/temp/vernagt13.jpg")

empty = np.zeros_like(sky[:,:,0], dtype=bool)

skymask = np.copy(empty)
skymask[sky[:,:,0]<125] = True

clearmask = np.copy(empty)
clearmask[(288,271,229,230,244,252,245),(210,500,627,732,857,1720,1344)] = True

classificationmask = np.ones_like(classification[:,:,0], dtype=bool)
classificationmask[classification[:,:,0] < 125] = False

# Check results
fig = plt.figure()
st = fig.suptitle("Check results", fontsize="x-large")

plt.subplot(2, 2, 1)
plt.imshow(skymask)
plt.title("Skymask")

plt.subplot(2, 2, 2)
plt.imshow(clearmask, interpolation="none")
plt.title("Clearmask")

plt.subplot(2, 2, 3)
plt.imshow(classificationmask, interpolation="none")
plt.title("Snowclassification")

plt.subplot(2, 2, 4)
plt.imshow(image, interpolation="none")
plt.title("Image")

plt.show()

#np.savez_compressed("C:/Master/settings/%s/settings.npz" %name, skymask = skymask, clearmask=clearmask, classification=classificationmask)

im = Image.fromarray(classificationmask.astype("uint8")*255)
im.save("C:/Master/settings/%s/classificationmask.jpg" %name)
