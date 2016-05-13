import numpy as np
from PIL import Image, ImageOps
from scipy.signal import argrelextrema
import os, time, bisect, sys, ShadowScript, SkyDetection
import Input_vernagtferner as Input
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import osgeo.gdal as gdal
from numpy import genfromtxt
plt.style.use('ggplot')

def ReadImage(image, img):
    # Open Reference Image

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


def movingaverage(values, window):
    weigths = np.repeat(1.0, window)/window
    sub = window%2
    add = (window - sub)/2
    add = [0]*add
    # including valid will REQUIRE there to be enough datapoints.
    # for example, if you take out valid, it will start @ point one,
    # not having any prior points, so itll be 1+0+0 = 1 /3 = .3333
    smas = np.convolve(values, weigths, 'valid')
    smas = np.insert(smas, 0, add)
    smas = np.insert(smas, len(smas), add)
    return smas


# load correspondence file
if os.path.exists(Input.CorrespondenceFile+ "cor.npz"):
    with np.load(Input.CorrespondenceFile+ "cor.npz") as data:
        cor = data['cor']
else:
    cor = np.loadtxt(Input.CorrespondenceFile, delimiter=",")
    np.savez_compressed(Input.CorrespondenceFile + "cor.npz", cor=cor)

# index of raster cell, which can be seen
cond = np.where(cor[7,:] == 1)[0]
view = cor[:,cond]
arrayview = np.zeros((9, view.shape[1]))
# All values minus one, matlab starts with 1
arrayview[:4,:] = view[:4,:]-1


# read dgm Info
with open(Input.DGM, "r") as fobj:
    lines = fobj.readlines()
    header = "".join(lines[:6])[:-1]
    ncols = float(lines[0].split()[1])
    nrows = float(lines[1].split()[1])
    cellsize = float(lines[4].split()[1])
    nodata = float(lines[5].split()[1])

# Easy Cloud Cover algoritm
#BlueSkyMask = SkyDetection.SkyDetection(Input.CorrespondenceImage)

# plt.imshow(BlueSkyMask)
# plt.colorbar()
# plt.show()

########################################################################################################################
# Iterate over all Images in specific Folder

Images = os.listdir(Input.ImagesPath)

for element in Images:
    ImageInfo = element[:-4]
    #if os.path.exists(os.path.join(Input.SnowClassFolder,"SC%s.txt" %ImageInfo)):
    #if element[-4:] == ".jpg":
    if element == "2014-10-29_08-30_0.jpg":
        ImagePath = os.path.join(Input.ImagesPath, element)
        print "[+] Image:", ImageInfo
        print("[+] Status: %.2f Prozent" %(float(Images.index(element))/float(len(Images))*100))


        img = Image.open(ImagePath)

        # Autocontrast
        #img = ImageOps.autocontrast(img, cutoff=2)

        # Clean specific Values of Image
        arrayview[4:9,:] = 0

        # Get R,G,B arrays from Image
        r,g,b = ReadImage(ImagePath, img)

        # plt.imshow(img)
        # plt.scatter((r.shape[1]/2.), (r.shape[0]/2.))
        # plt.xlim(xmin=0, xmax=r.shape[1])
        # plt.ylim(ymax=0, ymin=r.shape[0])
        # plt.show()

        # fill arrayview with RGB Values
        arrayview[4] = r[(arrayview[3].astype(int)), (arrayview[2].astype(int))]
        arrayview[5] = g[(arrayview[3].astype(int)), (arrayview[2].astype(int))]
        arrayview[6] = b[(arrayview[3].astype(int)), (arrayview[2].astype(int))]

        ################################################################################################################

        # Easy Cloud Cover algoritm
        #CloudCover = SkyDetection.CloudCover(BlueSkyMask, ImagePath)

        # Image Shadow Detection
        if Input.ShadowDetection:
            #if CloudCover < 25:

            ShadowPath = ShadowScript.saga_shadow(ImageInfo)

            ShadowFile = gdal.Open(ShadowPath)
            ShadowRaster = ShadowFile.GetRasterBand(1).ReadAsArray()

            # Reclass Raster Shadow=1 NoShadow=0
            ShadowRaster[ShadowRaster == 0] = 1
            ShadowRaster[ShadowRaster != 1] = 0

            arrayview[8] = ShadowRaster[arrayview[0].astype(int),arrayview[1].astype(int)]

            shadow_view = np.full((nrows,ncols), nodata, dtype=float)
            shadow_view[arrayview[0].astype(int),arrayview[1].astype(int)] = arrayview[8]

            shadow_viewName = Input.ShadowRastFolder + "SH" + ImageInfo + ".txt"
            np.savetxt(shadow_viewName, shadow_view, header=header, fmt="%.2f", comments="")

            if Input.ColorCorrection:
                ShadowScript.ShadowColorCorrection(arrayview)

            if Input.ShadowAsImage:
                # ShadowImage for Presentation and Visualization
                ShadowScript.ShadowToImage(arrayview, r, ImageInfo, img)

        ################################################################################################################
        # Method for Classifiying Snow

        hist = np.histogram(arrayview[6,:], bins=np.arange(0, 256))
        maverage = movingaverage(hist[0], 5)
        lmin = argrelextrema(maverage, np.less)[0]
        index = bisect.bisect(lmin, 127)

        if index < len(lmin):
            snowpixel = lmin[index]
        else:
            snowpixel = 127

        if Input.SnowMethod == 1:
            ## Corripio Snow Detection
            isnow1 = np.where(arrayview[6, :] >= snowpixel)[0]

        elif Input.SnowMethod == 2:

            rgb = arrayview[4:7, :].T

            rgb_centnorm = (rgb-np.tile(np.mean(rgb, axis=0),(rgb.shape[0],1)))/np.tile(np.std(rgb, axis=0, ddof=1),(rgb.shape[0],1))
            dummyU,dummyS,rgb_pc = np.linalg.svd(rgb_centnorm, full_matrices=False)
            rgb_pc = rgb_pc.T
            rgb_sc=rgb_centnorm.dot(rgb_pc)
            del rgb_centnorm,dummyU,dummyS

            pca = np.divide((rgb_sc - np.min(rgb_sc)), np.max(rgb_sc) - np.min(rgb_sc))
            del rgb_sc

            isnow1 = np.where(arrayview[6, :] >= snowpixel)[0]

            isnow2 = np.logical_and(np.logical_and(pca[:,2] <pca[:,1], rgb[:,2] >= Input.tbl), rgb[:, 2] < snowpixel).nonzero()[0]

            irock = (~(((pca[:, 2] < pca[:, 1]) & (rgb[:,2] >= Input.tbl)) | (rgb[:,2]>=snowpixel)) & (rgb[:,2]<rgb[:,0])).nonzero()[0]

            i5050 = (~(((pca[:,2] <pca[:,1]) & (rgb[:,2] >= Input.tbl)) | (rgb[:,2]>=snowpixel)) & (rgb[:,2]>=rgb[:,0])).nonzero()[0]

            del rgb

        arrayview[7,isnow1] = 1

        if Input.SnowMethod == 2:
            arrayview[7,isnow2] = 2
            arrayview[7,irock] = -1
            maxi = snowpixel
            if len(i5050) == 0:
                continue
            mini = np.max([(np.min(arrayview[6,i5050])-1), (Input.tbl - 1)])
            arrayview[7,i5050] = 1 / (maxi-mini)*(arrayview[6,i5050]-mini)
            arrayview[7,arrayview[7,:]>1] = 1
            arrayview[7,arrayview[7,:]<0] = 0



        ################################################################################################################
        # Create Raster SnowClassified for specific Image

        raster = np.full((nrows,ncols), nodata, dtype=float)
        raster[arrayview[0].astype(int),arrayview[1].astype(int)] = arrayview[7]

        SnowRasterName = Input.SnowClassFolder + "SC" + ImageInfo + ".txt"
        np.savetxt(SnowRasterName, raster, header=header, fmt="%.2f", comments="")


        ################################################################################################################
        # Plots

        if Input.plot1:
            # PLOT for Color Histogram
            plt.hist(arrayview[6, :], bins=255, facecolor="#87C4ED", edgecolor='none')
            plt.plot(hist[1][:-1], maverage, color="#2408C2", lw=2)
            plt.axvline(127, color='r', linestyle='dashed', linewidth=2)
            plt.axvline(snowpixel, color='g', linestyle='dashed', linewidth=2)
            #plt.ylim(ymax=6000)
            plt.xlabel("Digital Number")
            plt.ylabel("Number of pixels")
            plt.title('Histogram of blue values')
            plt.show()

        if Input.plot2:
            dgm = np.loadtxt(Input.DGM, skiprows=6)
            height = dgm[arrayview[0].astype(int),arrayview[1].astype(int)].astype(int)
            min,max = np.min(height),np.max(height)
            height_dif = int(round((max-min)/10))
            fig = plt.figure()
            ax = fig.add_subplot(111)
            x = plt.hist(height[arrayview[7,:]==1], bins=height_dif)
            print x
            vals = ax.get_yticks()
            ax.set_yticklabels(['{:.2f} $km^2$'.format(x/100) for x in vals])
            plt.xlabel("in meter")
            plt.grid(True)
            plt.show()

        if Input.plot3:
            import plots.colormap as cm
            fig = plt.figure(figsize=(30,10))
            ax = fig.add_subplot(121)
            plt.imshow(img)
            plt.xlim(xmin=np.min(arrayview[2,:]), xmax=np.max(arrayview[2,:]))
            plt.ylim(ymin=np.max(arrayview[3,:]),ymax=np.min(arrayview[3,:])-200)
            plt.xticks([])
            plt.yticks([])

            ax = fig.add_subplot(122)
            plt.imshow(img)
            plt.scatter(arrayview[2,:],arrayview[3,:], c=arrayview[7,:], s=1, lw=0, cmap=cm.redgreen)
            plt.xlim(xmin=np.min(arrayview[2,:]), xmax=np.max(arrayview[2,:]))
            plt.ylim(ymin=np.max(arrayview[3,:]),ymax=np.min(arrayview[3,:])-200)
            plt.xticks([])
            plt.yticks([])
            #plt.colorbar()
            plt.tight_layout()
            if Input.saveplot3:
                plt.savefig(os.path.join(Input.pathplot3, "%s.jpg" % ImageInfo))
            else:
                plt.show()
            plt.close()

        # save correspondence
        np.savetxt(Input.CorrondenceFolder + "C_" + ImageInfo + ".txt", arrayview, fmt="%.2f")


# for file in os.listdir(Input.Temp):
#     try:
#         os.remove(os.path.join(Input.Temp,file))
#     except:
#         pass










