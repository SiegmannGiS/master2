import os
import Input_vernagtferner as Input
import numpy as np
from scipy import misc
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from math import *
import calendar
from astral import Astral

File = Input.DGM.split("/")[-1][:-4]
SAGAFile = File+".sgrd"
# Import DGM Grid
if not os.path.exists(Input.Temp+SAGAFile):
    os.system("\"C:\Program Files (x86)\SAGA-GIS\saga_cmd\" io_grid 1 -GRID %s%s -FILE %s" % (Input.Temp, SAGAFile, Input.DGM))


def summertime(year):
    laststart_sunday = max(week[-1] for week in calendar.monthcalendar(year, 3))
    lastend_sunday = max(week[-1] for week in calendar.monthcalendar(year, 10))
    return datetime(year=year,month=3,day=laststart_sunday, hour=2),datetime(year=year,month=10,day=lastend_sunday, hour=3)


def saga_shadow(ImageInfo):
    # Split ImageInfo
    ImageInfo = "_".join(ImageInfo.split("_")[:2])
    date = datetime.strptime(ImageInfo,"%Y-%m-%d_%H-%M")
    Minute = "%i" % (int(date.minute) / 60. * 100.)
    Moment = "%s.%s" % (date.hour, Minute)

    B = 360 / 365 * (date.timetuple().tm_yday - 81)
    EoT = 9.81 * sin(radians(2 * B)) - 7.53 * cos(radians(B)) - 1.5 * sin(radians(B))

    if Input.summertime:
        if summertime(date.year)[0] < date < summertime(date.year)[1]:
            LST = (float(Moment) + (EoT / 60.)) -1
            utc = date - timedelta(hours=2)
            print("summertime")
        else:
            LST = (float(Moment) + (EoT / 60.))
            utc = date - timedelta(hours=1)
            print("wintertime")
    else:
        LST = (float(Moment) + (EoT / 60.))
        utc = date - timedelta(hours=1)

    print utc
    day = datetime.strftime(date, "%m/%d/%Y")
    #LST = LST -0.25
    #Calculate solar radiation
    os.system("\"C:\Program Files (x86)\SAGA-GIS\saga_cmd\" ta_lighting 2 "
              "-GRD_DEM %s "
              "-GRD_DIRECT %s "
              "-DAY %s "
              "-MOMENT %s "
              "-PERIOD 0 "
              "-LATITUDE %s "
              "-METHOD 3"
              % (os.path.join(Input.Temp, SAGAFile), os.path.join(Input.Temp, "Direct.sgrd"), day, LST, Input.Latitude))


    # os.system("\"C:\Program Files (x86)\SAGA-GIS\saga_cmd\" io_grid 0 -GRID %s%s -FILE %sShadow.txt"
    #           % (Input.Temp,"Direct.sgrd",Input.Temp))

    return os.path.join(Input.Temp, "Direct.sdat"),utc


def ShadowToImage(arrayview, r, Imageinfo, img):
    "ShadowImage for Presentation and Visualization"
    # Image Creation
    ShadowImage = np.full(r.shape, 0.5)

    ShadowImage[arrayview[3].astype(int)-1,arrayview[2].astype(int)-1] = arrayview[8].astype(int)

    ShadowImage[ShadowImage==1] = 2
    ShadowImage[ShadowImage==0] = 1
    ShadowImage[ShadowImage==2] = 0

    plt.imshow(img)
    plt.scatter(arrayview[2, :], arrayview[3, :], c=arrayview[8, :], s=2, lw=0, cmap="Greys")
    plt.xlim(xmin=np.min(arrayview[2, :]), xmax=np.max(arrayview[2, :]))
    plt.ylim(ymin=np.max(arrayview[3, :]), ymax=np.min(arrayview[3, :]) - 200)
    plt.xticks([])
    plt.yticks([])
    # plt.colorbar()
    #plt.tight_layout()

    plt.show()
    #plt.savefig("%s.jpg" % Imageinfo)
    plt.close()

def ShadowToImage2(arrayview, r, img):
    plt.imshow(img)
    plt.scatter(arrayview[2, :], arrayview[3, :], c=arrayview[8, :], s=2, lw=0, cmap="bwr")
    plt.show()


    print "done"


def ShadowColorCorrection(arrayview):

    rgb2double = (arrayview[4:7].astype(float))/255
    shadow_core = np.copy(arrayview[8,:])
    lit_core = 1*np.logical_not(shadow_core)

    # averaging pixel intensities in the shadow/lit areas
    shadowavg_red = np.sum(np.sum(rgb2double[0,:]*shadow_core)) / np.sum(np.sum(shadow_core))
    shadowavg_green = np.sum(np.sum(rgb2double[1,:]*shadow_core)) / np.sum(np.sum(shadow_core))
    shadowavg_blue = np.sum(np.sum(rgb2double[2,:]*shadow_core)) / np.sum(np.sum(shadow_core))

    litavg_red = np.sum(np.sum(rgb2double[0,:]*lit_core)) / np.sum(np.sum(lit_core))
    litavg_green = np.sum(np.sum(rgb2double[1,:]*lit_core)) / np.sum(np.sum(lit_core))
    litavg_blue = np.sum(np.sum(rgb2double[2,:]*lit_core)) / np.sum(np.sum(lit_core))

    if Input.ColorOption == 1:
        # additive shadow removal

        # computing colour difference between the shadow/lit areas
        diff_red = litavg_red - shadowavg_red
        diff_green = litavg_green - shadowavg_green
        diff_blue = litavg_blue - shadowavg_blue

        # adding the difference to the shadow pixels
        arrayview[4,:] = ((rgb2double[0,:] + shadow_core*diff_red)*255).astype(np.uint8)
        arrayview[5,:] = ((rgb2double[1,:] + shadow_core*diff_green)*255).astype(np.uint8)
        arrayview[6,:] = ((rgb2double[2,:] + shadow_core*diff_blue)*255).astype(np.uint8)

    if Input.ColorOption == 2:
        # basic, light model based shadow removal

        # computing ratio of shadow/lit area luminance
        ratio_red = litavg_red/shadowavg_red
        ratio_green = litavg_green/shadowavg_green
        ratio_blue = litavg_blue/shadowavg_blue

        # multiplying the shadow pixels with the ratio for the correction
        arrayview[4,:] = ((rgb2double[0,:]*lit_core + shadow_core*ratio_red*rgb2double[0,:])*255).astype(np.uint8)
        arrayview[5,:] = ((rgb2double[1,:]*lit_core + shadow_core*ratio_green*rgb2double[1,:])*255).astype(np.uint8)
        arrayview[6,:] = ((rgb2double[2,:]*lit_core + shadow_core*ratio_blue*rgb2double[2,:])*255).astype(np.uint8)

    return arrayview
