from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from skimage.color import rgb2hsv
from skimage.io import imread

def percentage(part, whole):
    return 100 * float(part)/float(whole)

# clear = np.load("C:/master/settings/vernagtferner/clear.npz")["clear"]
# sky = np.load("C:/master/settings/vernagtferner/sky.npz")["sky"]
img = imread("C:\Master\images\Vernagtferner/2014-10-04_07-30.jpg")

hsv = rgb2hsv(img)

skychannel = (hsv[:, :, 1] * 100).astype(int)
clearchannel = (hsv[:, :, 2] * 100).astype(int)
#
# print skychannel[sky] < 30
# print clearchannel[clear] < 50
#
# print np.sum(sky)
# print np.sum(skychannel[sky] < 30)
# print int(percentage(np.sum(skychannel[sky] < 30),np.sum(sky)))

#
# fig = plt.figure()
# st = fig.suptitle("HSV color model", fontsize="x-large")
#
# plt.subplot(2, 2, 1)
# plt.imshow(skychannel, vmin=0, vmax=100, cmap=cm.get_cmap("Greys"), interpolation="none")
# plt.title("Saturation")
# plt.colorbar()
#
# plt.subplot(2, 2, 2)
# plt.imshow(clearchannel, vmin=0, vmax=100, cmap=cm.get_cmap("Greys"), interpolation="none")
# plt.title("Value")
# plt.colorbar()
#
# plt.subplot(2, 2, 3)
# #test = img[:,:,2]/255.*100.
# test = img[:,:,2]
# plt.imshow(test, interpolation="none", cmap=cm.get_cmap("Greys"))
# plt.colorbar()
# plt.title("Image")
#
# plt.subplot(2, 2, 4)
# plt.imshow((hsv[:, :, 0] * 360).astype(int), vmin=0, vmax=360, cmap=cm.get_cmap("hsv"), interpolation="none")
# plt.title("Hue")
# plt.colorbar()
# plt.tight_layout()
# plt.show()

from mpl_toolkits.axes_grid1 import make_axes_locatable

fig = plt.figure()
fig.suptitle("HSV color model", fontsize="x-large")
plt.subplot(1,2,1)
plt.imshow(img)

plt.subplot(1,2,2)
ax = plt.gca()
im = ax.imshow((hsv[:, :, 0] * 360).astype(int), vmin=0, vmax=360, cmap=cm.get_cmap("hsv"), interpolation="none")
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)

plt.colorbar(im, cax=cax)
plt.tight_layout()h
plt.show()