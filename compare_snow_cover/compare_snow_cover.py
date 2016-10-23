import numpy as np
import os
import datetime
from amundsen.netcdf import ReadNetCDF
from tools import ascii
from skimage.transform import downscale_local_mean


def FileToDate(fname, string):
    date = datetime.datetime.strptime(fname, string)
    return date


def resample_binary_majority(raster, factor):
    resampled = downscale_local_mean(raster, factors=factor)
    resampled_bool = np.zeros_like(resampled)
    resampled_bool[resampled >= 0.5] = 1

    mask = np.ones_like(resampled)
    if raster.shape[0] % factor[0] != 0:
        y = (factor[0] - (raster.shape[0] % factor[0])) * -1
        mask[y:,:] = 0
    if raster.shape[1] % factor[1] != 0:
        x = (factor[1] - (raster.shape[1] % factor[1])) * -1
        mask[:,x:] = 0
    # Create mask

    mask = mask.astype(bool)
    return resampled_bool, mask


def PreparePractise(PractiseDict, dateobj, threshold):
    day = dateobj.date()
    RastPerDay = [element for key, element in PractiseDict.iteritems() if key.date() == dateobj.date()]
    raster = np.array([ascii.read_ascii(path)[1] for path in RastPerDay])
    meanRast = np.mean(raster, axis=0)
    meanRastBool = np.zeros_like(meanRast)
    meanRastBool[meanRast >= threshold] = 1
    mask = ~np.isnan(meanRast)

    return [day, meanRastBool, mask]


def PrepareLandsat(LandsatDict, dateobj):
    day = dateobj.date()
    RastPerDay = [element for key, element in LandsatDict.iteritems() if key.date() == dateobj.date()][0]
    cloud = ascii.read_ascii(RastPerDay["cloud"])[1].astype(bool)
    cirrus = ascii.read_ascii(RastPerDay["cirrus"])[1].astype(bool)
    snow = ascii.read_ascii(RastPerDay["snow"])[1]
    mask = ~(cirrus | cloud)

    return [day, snow, mask]


def PrepareAmundsen(AmundsenDict, dateobj):
    day = dateobj.date()
    amundsen_date = datetime.datetime(day.year, day.month, day.day, 12, 0)
    rast = AmundsenDict[str(day.year)].get_array_from_datetime(amundsen_date, "swe")
    snow = np.zeros_like(rast)
    snow[rast > 0] = 1
    mask = ~np.isnan(rast)

    return [day, snow, mask]


class CompareSnowCover(object):
    def __init__(self,site):
        self.site = site
        self.PractiseDirName = "C:\Master\snowdetection/%s" % site
        self.LandsatDirName = "C:\Master\landsat/%s/landsat" % site
        self.AmundsenDirName = "C:\Master/amundsen/%s/results" % site
        self.PractiseDict = {}
        self.LandsatDict = {}
        self.AmundsenDict = {}

    def GetPractiseFiles(self):
        for fname in os.listdir(self.PractiseDirName):
            date = FileToDate(fname, "SC%Y-%m-%d_%H-%M.txt")
            self.PractiseDict[date] = os.path.join(self.PractiseDirName, fname)

    def GetLandsatFiles(self):
        dates = list(set([x.split("_")[1] for x in os.listdir(self.LandsatDirName)]))

        for date in dates:
            matching = [s for s in os.listdir(self.LandsatDirName) if date in s]
            for fname in matching:
                if ("snow" in fname) and (fname.split(".")[-1] == "asc"):
                    date = FileToDate(fname, 'lc8_%Y-%m-%d_snow.asc')
                    snow = os.path.join(self.LandsatDirName, fname)
                elif ("cirrus" in fname) and (fname.split(".")[-1] == "asc"):
                    cirrus = os.path.join(self.LandsatDirName, fname)
                elif ("cloud" in fname) and (fname.split(".")[-1] == "asc"):
                    cloud = os.path.join(self.LandsatDirName, fname)

            self.LandsatDict[date] = {"snow": snow, "cirrus": cirrus, "cloud": cloud}


    def GetNcObjects(self):
        for year in os.listdir(self.AmundsenDirName):
            self.AmundsenDict[year] = ReadNetCDF(os.path.join(self.AmundsenDirName, year, "OutputFields_%s.nc" % year))


if __name__ == "__main__":
   test = np.arange(36).reshape((6,6))
   test2 = resample_binary_majority(test, (2,2))

