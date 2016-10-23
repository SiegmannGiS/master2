import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import PiecewiseAffineTransform, warp
from skimage import data
from scipy.ndimage import imread
import os
import numpy as np
from tools import ascii

filepath = "C:\Master\settings/astental\correspondence.txt"
path = "C:\Master\settings/astental"
headinfo = ascii.read_ascii("C:\Master\settings/astental/dgm_astental.asc")[0]
correspondence = np.loadtxt(filepath, delimiter=",")

image = imread("C:\Master\images/astental/2014-09-24_12-00.jpg")
rows, cols = image.shape[0], image.shape[1]

src = np.transpose(correspondence[:2,:].astype(int))[::1000]
y = (correspondence[0] * headinfo[-2]) + headinfo[3]
x = (correspondence[1] * headinfo[-2]) + headinfo[2]
dst = np.transpose(np.array([x,y]))[::1000]


tform = PiecewiseAffineTransform()
tform.estimate(src, dst)

out = warp(image, tform)
print(out)
fig, ax = plt.subplots()
ax.imshow(out)
ax.plot(tform.inverse(src)[:, 0], tform.inverse(src)[:, 1], '.b')
plt.show()