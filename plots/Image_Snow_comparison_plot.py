import os
import matplotlib.pyplot as plt
import numpy as np
import plots.colormap as cm

path = "C:\Master"
area = "vernagtferner14-16"

with np.load(os.path.join(path,"settings",area,"correspondence.txtcor.npz")) as fobj:
    cor = fobj["cor"]


print cor[:,:3]

for i, image in enumerate(sorted(os.listdir(os.path.join(path,"images",area)))):
    if i == 0:
        imagepath = os.path.join(path,"images",area,image)
        snowcover = os.path.join(path,"snowdetection",area,"SC%s.txt" %image[:-4])
        print snowcover

        # plot


        # plt.figure(1)
        # plt.imshow(img)
        # plt.scatter(arrayview[2, :], arrayview[3, :], c=arrayview[7, :], s=1, lw=0, cmap=cm.redgreen)
        # plt.xlim(xmin=np.min(arrayview[2, :]), xmax=np.max(arrayview[2, :]))
        # plt.ylim(ymin=np.max(arrayview[3, :]), ymax=np.min(arrayview[3, :]) - 200)
        # plt.xticks([])
        # plt.yticks([])
        # plt.colorbar()
        # plt.show()