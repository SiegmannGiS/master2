import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import os
import math
from matplotlib import cm
from scipy import ndimage
plt.style.use('ggplot')

def roundup(x):
    return int(math.ceil(x / 100.0)) * 100

def round_down(num):
    return num - (num%100)


# Settings
snowpath = "C:\Master\snowdetection/vernagtferner14-16"
dgm = "C:\Master\settings/vernagtferner14-16/dgm_vernagtferner.txt"
glacier = np.loadtxt("C:\Master\settings/vernagtferner14-16/glacierarea.txt",skiprows=6)

# Period to look at
viewperiod = (datetime(year=2014,month=10,day=10),datetime(year=2015, month=4, day=30))

# gaussian filter
gaussian = True
std = (1)

# nodata
nodata = -9999
cellsize = 5

# height steps
heightdistribution = False
steps = 50

# load dgm
dgm = np.loadtxt(dgm,skiprows=6)
dgm[dgm == nodata] = np.nan

# read snowfiles
if not os.path.exists("data.npz"):
    datalist = []
    for i,file in enumerate(sorted(os.listdir(snowpath))):
        if i > 0:
            print file
            filepath = os.path.join(snowpath,file)
            date = datetime.strptime("-".join(file.split("_")[:2]), "SC%Y-%m-%d-%H-%M")
            data = np.loadtxt(filepath,skiprows=6)
            data[data == nodata] = np.nan

            # DGM in viewing field of camera

            heights = dgm[~np.isnan(data)]

            # get bins
            minheight = round_down(np.min(heights))
            maxheight = roundup(np.max(heights))
            bins = (maxheight-minheight)/steps

            # Choose Snowpixel
            snow = dgm[data > 0]
            histheight = np.histogram(heights, bins=bins, range=[minheight,maxheight])
            histsnow = np.histogram(snow, bins=bins, range=[minheight,maxheight])

            #height plot
            if heightdistribution:
                if i == 0:
                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    plt.hist(heights, bins=bins, range=[minheight,maxheight], color="#525252", label="DTM")
                    glacier[glacier == nodata] = np.nan
                    plt.hist(glacier.flatten(), bins=bins, range=[minheight, maxheight], color="#508eff", alpha=0.5, label="Glacier area")
                    ax.set_yticklabels(['{:.2f}'.format((x*25)/1000000) for x in ax.get_yticks()])
                    plt.ylabel("Area per 50 m altitude level [$km^2$]")
                    plt.xlabel("Altitude level [m]")
                    legend = plt.legend(shadow=False, fancybox=False, title="Height distribution")
                    frame = legend.get_frame()
                    frame.set_edgecolor("black")
                    plt.tight_layout()
                    plt.savefig("height_distribution.jpeg")
                    plt.show()

            prozentflaeche = histsnow[0].astype(float)/histheight[0].astype(float)*100.

            xticks = range(int(np.min(histsnow[1])+50),int(np.max(histsnow[1])+50),steps)
            datalist.append([[date]*len(xticks),xticks,prozentflaeche])

    # comment out for plotting height distribution
    np.savez_compressed("data.npz", data=datalist)


with np.load("data.npz") as fobj:
    data = fobj["data"]


data = np.array([data[i,:] for i,x in enumerate(data[:,0,0]) if viewperiod[0] < x < viewperiod[1]])
data = np.array([data[i,:] for i,x in enumerate(data[:,0,0]) if x.hour == 11])



x = range(len(data[:,0]))
y = data[1,1]
z = data[:,2]


x = np.array([x]*len(data[0,0]))
y = np.rot90(np.array([y]*len(data)))
z = np.array(np.rot90(z).astype(int))

if gaussian:
    if not std:
        std = np.std(z)
    print std
    z = ndimage.gaussian_filter1d(z, std, 1)


fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=True)

print data[:,0,0][x]
ax.set_xticklabels([datetime.strftime(data[:,0,0][x], "%d.%m.%Y") for x in ax.get_xticks()[:-1]])
ax.tick_params(labelsize=10)

plt.ylabel("Height level [m]")
ax.set_zlabel("Snow percentage [%]")
plt.tight_layout()
plt.show()




