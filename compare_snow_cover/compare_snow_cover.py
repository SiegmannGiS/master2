import numpy as np
import os, sys
import datetime
from amundsen.netcdf import ReadNetCDF
from tools import ascii
from skimage.transform import downscale_local_mean
import matplotlib.pyplot as plt
import pandas as pd


def mask_nan(result):
    rast = result[1]
    mask = result[2]
    rast[~mask] = np.nan
    return rast


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
    return [factor, resampled_bool, mask]


def compare_datetime(site, model, obs):
    site = CompareSnowCover(site)

    sitedict = {"landsat": [site.GetLandsatFiles, PrepareLandsat, site.LandsatDict],
                "amundsen": [site.GetNcObjects, PrepareAmundsen, site.AmundsenDict],
                "practise": [site.GetPractiseFiles, PreparePractise, site.PractiseDict]}

    sitedict[model][0]()
    sitedict[obs][0]()


    if model == "amundsen" or obs == "amundsen":
        amundsen_dates = []
        [amundsen_dates.extend(y) for y in [sitedict["amundsen"][2][x].get_time_list() for x in sitedict["amundsen"][2]]]
        amundsen_dates = sorted(list(set([x.date() for x in amundsen_dates])))

        if model == "landsat" or model == "practise":
            liste = sorted(list(set([x.date() for x in sitedict[model][2].keys()])))
        if obs == "landsat" or obs == "practise":
            liste = sorted(list(set([x.date() for x in sitedict[obs][2].keys()])))

        result = set(amundsen_dates) & set(liste)
    else:
        modellist = sorted(list(set([x.date() for x in sitedict[model][2].keys()])))
        obslist = sorted(list(set([x.date() for x in sitedict[obs][2].keys()])))
        result = set(modellist) & set(obslist)

    return list(result)


def PreparePractise(PractiseDict, dateobj, threshold=0.5):
    if not isinstance(dateobj, datetime.date):
        day = dateobj.date()
    else:
        day = dateobj
    print("practise", day)
    RastPerDay = [element for key, element in PractiseDict.iteritems() if key.date() == day]
    raster = np.array([ascii.read_ascii(path)[1] for path in RastPerDay])
    meanRast = np.mean(raster, axis=0)
    meanRastBool = np.zeros_like(meanRast)
    meanRastBool[meanRast >= threshold] = 1
    mask = ~np.isnan(meanRast)

    return [day, meanRastBool, mask]


def PrepareLandsat(LandsatDict, dateobj):
    if not isinstance(dateobj, datetime.date):
        day = dateobj.date()
    else:
        day = dateobj
    print("landsat", day)
    RastPerDay = [element for key, element in LandsatDict.iteritems() if key.date() == day][0]
    cloud = ascii.read_ascii(RastPerDay["cloud"])[1].astype(bool)
    cirrus = ascii.read_ascii(RastPerDay["cirrus"])[1].astype(bool)
    snow = ascii.read_ascii(RastPerDay["snow"])[1]
    mask = ~(cirrus | cloud)

    return [day, snow, mask]


def PrepareAmundsen(AmundsenDict, dateobj):
    if not isinstance(dateobj, datetime.date):
        day = dateobj.date()
    else:
        day = dateobj
    print("amundsen", day)
    amundsen_date = datetime.datetime(day.year, day.month, day.day, 12, 0)
    rast = AmundsenDict[str(day.year)].get_array_from_datetime(amundsen_date, "swe")
    snow = np.zeros_like(rast)
    snow[rast >= 1] = 1
    mask = ~np.isnan(rast)

    return [day, snow, mask]


def create_dataset(site, model, obs, dateobject, mask=False, overwrite=False):
    if overwrite or not os.path.exists("data/stats_%s_%s_%s.csv" % (site, model, obs)):
        sitename = site
        site = CompareSnowCover(site)

        sitedict = {"landsat": [site.GetLandsatFiles, PrepareLandsat, site.LandsatDict],
                    "amundsen": [site.GetNcObjects, PrepareAmundsen, site.AmundsenDict],
                    "practise": [site.GetPractiseFiles, PreparePractise, site.PractiseDict]}

        sitedict[model][0]()
        sitedict[obs][0]()

        if all([isinstance(x, datetime.date) for x in dateobject]):
            dateobj = dateobject
        else:
            sys.exit("dateobject is not valid")

        stats_dict = {}

        for date in sorted(dateobj):
            print(date)
            res_model = sitedict[model][1](sitedict[model][2], date)
            res_obs = sitedict[obs][1](sitedict[obs][2], date)

            if model == "landsat":
                res_obs[1] = np.array(resample_binary_majority(res_obs[1], (3,3)))[1]
                res_obs[2] = np.array(resample_binary_majority(res_obs[2], (3, 3)))[1].astype(bool)
            elif obs == "landsat":
                res_model[1] = np.array(resample_binary_majority(res_model[1], (3,3)))[1]
                res_model[2] = np.array(resample_binary_majority(res_model[2], (3, 3)))[1].astype(bool)

            if site == "vernagtferner":
                if model == "landsat":
                    res_obs[1] = np.array(resample_binary_majority(res_obs[1], (2, 2)))[1]
                    res_obs[2] = np.array(resample_binary_majority(res_obs[2], (2, 2)))[1].astype(bool)
                elif obs == "landsat":
                    res_model[1] = np.array(resample_binary_majority(res_model[1], (2, 2)))[1]
                    res_model[2] = np.array(resample_binary_majority(res_model[2], (2, 2)))[1].astype(bool)

            if type(mask) != bool:
                stats_object = Verification(res_model[1], res_obs[1], res_model[2], res_obs[2], mask)
            else:
                stats_object = Verification(res_model[1], res_obs[1], res_model[2], res_obs[2])

            stats_dict[date] = [stats_object.ACC(),stats_object.CSI(),stats_object.BIAS()]

        stats_df = pd.DataFrame(stats_dict.values(), index=stats_dict.keys(), columns=["ACC", "CSI", "BIAS"])
        stats_df = stats_df.sort_index()

        stats_df.name = "stats_%s_%s_%s" % (sitename, model, obs)
        stats_df.to_csv("data/stats_%s_%s_%s.csv" % (sitename, model, obs), index_label="date")


    else:
        stats_df = pd.read_csv("data/stats_%s_%s_%s.csv" % (site, model, obs))

    return stats_df


class CompareSnowCover(object):
    def __init__(self,site):
        self.site = site
        self.PractiseDirName = "C:\Master\snowdetection/%s" % site
        self.LandsatDirName = "C:\Master\landsat/%s/landsat" % site
        self.AmundsenDirName = "C:\Master/amundsen/%s/results" % site
        self.gispath = "C:\Master/GIS/%s" % site
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


class Verification(object):
    def __init__(self, SimRast, ObsRast, SimMask=0, ObsMask=0, AreaMask=0):
        """

        :param SimRast: Simulation Raster
        :param ObsRast: Observation Raster
        :param SimMask: Mask for Simulation, must not be declared (e.g.  areas which shall not taken into account)
        :param ObsMask: Mask for observation, must not be declared (e.g. Cloudmask for landsat data)
        :param AreaMask: Mask for investigating special areas, must not be declared (e.g. Areas over 1000 meter height)
        """

        assert SimRast.shape == ObsRast.shape, "Rasters does not have same dimension"

        if type(SimMask) == int:
            SimMask = np.ones_like(SimRast).astype(bool)
        if type(ObsMask) == int:
            ObsMask = np.ones_like(ObsRast).astype(bool)
        if type(AreaMask) == int:
            AreaMask = np.ones_like(SimRast).astype(bool)

        assert SimMask.shape == ObsMask.shape, "Masks does not have same dimension"
        assert SimMask.shape == AreaMask.shape, "Masks does not have same dimension"
        mask = SimMask & ObsMask & AreaMask

        # Create contingency parts
        n11 = float(np.nansum((SimRast == 1) & (ObsRast == 1) & mask))
        n10 = float(np.nansum((SimRast == 1) & (ObsRast == 0) & mask))
        n01 = float(np.nansum((SimRast == 0) & (ObsRast == 1) & mask))
        n00 = float(np.nansum((SimRast == 0) & (ObsRast == 0) & mask))

        n1x = float(n11 + n10)
        n0x = float(n01 + n00)
        nx1 = float(n11 + n01)
        nx0 = float(n10 + n00)

        assert (n1x + n0x) == (nx1 + nx0), "Contigency Sum nxx is not valid"
        nxx = float(n1x + n0x)

        self.contingency = {"n11":n11, "n10":n10, "n01":n01, "n00":n00, "n1x":n1x, "n0x":n0x, "nx1":nx1, "nx0":nx0, "nxx":nxx}


    def ACC(self):
        """ The accuracy, ACC, is the number of correct forecasts for events and non-events divided by the
            total number of forecasts: """

        num = self.contingency
        acc = (num["n11"] + num["n00"]) / num["nxx"]
        return acc


    def BIAS(self):
        """ The BIAS score quantifies the relative frequency of predicted and observed events """

        num = self.contingency
        try:
            bias = num["n1x"] / num["nx1"]
        except ZeroDivisionError:
            bias = np.nan
        return bias


    def FAR(self):
        """ The false alarm ratio, FAR, indicates the fraction of event forecasts that were actually non-events.
            FAR is sensitive only to false predictions, and not to missed events """

        num = self.contingency
        far = num["n10"] / num["n1x"]
        return far


    def CSI(self):
        """ The critical success index CSI (Schaefer, 1990) is the number of correct event forecasts divided by
            the number of cases forecast and/or observed """

        num = self.contingency
        csi = num["n11"] / (num["nxx"] - num["n00"])
        return csi


    def HSS(self):
        """ The Heidke skill score
        HSS is a measure of correct forecasts; with random correct forecasts removed (i.e. forecasts
        expected to be correct by chance). The reference forecast in HSS is random chance F, subject to
        the constraint that marginal distributions of forecasts are the same as the marginal distributions of
        observations """

        num = self.contingency
        hss = (num["n11"] * num["n00"] - num["n01"] * num["n10"]) / ((num["nx1"] * num["n0x"] + num["n1x"] * num["nx0"]) / 2)
        f = ((num["n00"] + num["n10"]) * (num["n00"] + num["n01"]) + (num["n11"] + num["n10"]) * (num["n11"] + num["n01"])) / num["nxx"]
        return hss, f


class VerificationRaster(object):
    def __init__(self, site, model, obs, dateobject, mask=False, overwrite_npz=False):
        self.sitename = site

        if overwrite_npz or not os.path.exists("data/%s_%s-%s_VerificationRast.npz" %(site, model, obs)):

            site = CompareSnowCover(site)

            sitedict = {"landsat":[site.GetLandsatFiles, PrepareLandsat, site.LandsatDict],
                        "amundsen":[site.GetNcObjects, PrepareAmundsen, site.AmundsenDict],
                        "practise":[site.GetPractiseFiles, PreparePractise, site.PractiseDict]}

            sitedict[model][0]()
            sitedict[obs][0]()

            if dateobject == "model":
                dateobj = sorted(list(set([x.date() for x in sitedict[model][2].keys()])))
            elif dateobject == "obs":
                dateobj = sorted(list(set([x.date() for x in sitedict[obs][2].keys()])))
            elif all([isinstance(x, datetime.date) for x in dateobject]):
                dateobj = dateobject
            else:
                sys.exit("dateobject is not valid")

            observation = np.array([mask_nan(sitedict[obs][1](sitedict[obs][2], x)) for x in dateobj]) # if x in sitedict[arg][2].keys()
            simulation = np.array([mask_nan(sitedict[model][1](sitedict[model][2], x)) for x in dateobj]) # if x in sitedict[arg][2].keys()

            if model == "landsat":
                observation = np.array([mask_nan(resample_binary_majority(x, (3,3))) for x in observation])
            elif obs == "landsat":
                simulation = np.array([mask_nan(resample_binary_majority(x, (3,3))) for x in simulation])

            if site == "vernagtferner":
                if model == "landsat":
                    observation = np.array([mask_nan(resample_binary_majority(x, (2, 2))) for x in observation])
                elif obs == "landsat":
                    simulation = np.array([mask_nan(resample_binary_majority(x, (2, 2))) for x in simulation])

            simulation[np.isnan(observation)] = np.nan
            observation[np.isnan(simulation)] = np.nan

            # Create contingency parts
            n11 = np.nansum((simulation == 1) & (observation == 1), axis=0).astype(float)
            n10 = np.nansum((simulation == 1) & (observation == 0), axis=0).astype(float)
            n01 = np.nansum((simulation == 0) & (observation == 1), axis=0).astype(float)
            n00 = np.nansum((simulation == 0) & (observation == 0), axis=0).astype(float)

            n1x = n11 + n10
            n0x = n01 + n00
            nx1 = n11 + n01
            nx0 = n10 + n00
            nxx = n1x + n0x

            np.savez_compressed("data/%s_%s-%s_VerificationRast.npz" %(self.sitename, model, obs), n11=n11, n10=n10, n01=n01, n00=n00, n1x=n1x,
                                n0x=n0x, nx1=nx1, nx0=nx0, nxx=nxx)

            self.contingency = {"n11":n11, "n10":n10, "n01":n01, "n00":n00, "n1x":n1x, "n0x":n0x, "nx1":nx1, "nx0":nx0,
                                "nxx":nxx}
            if mask:
                site.mask

        else:
            npz = np.load("%s_%s-%s_VerificationRast.npz" %(self.sitename, model, obs))

            self.contingency = {"n11": npz["n11"], "n10": npz["n10"], "n01": npz["n01"], "n00": npz["n00"],
                                "n1x": npz["n1x"], "n0x": npz["n0x"], "nx1": npz["nx1"], "nx0": npz["nx0"],
                                "nxx": npz["nxx"]}

    def ACC(self):
        """ The accuracy, ACC, is the number of correct forecasts for events and non-events divided by the
            total number of forecasts: """

        num = self.contingency
        acc = (num["n11"] + num["n00"]) / num["nxx"]
        return acc

    def BIAS(self):
        """ The BIAS score quantifies the relative frequency of predicted and observed events """

        num = self.contingency
        bias = num["n1x"] / num["nx1"]
        return bias

    def FAR(self):
        """ The false alarm ratio, FAR, indicates the fraction of event forecasts that were actually non-events.
            FAR is sensitive only to false predictions, and not to missed events """

        num = self.contingency
        far = num["n10"] / num["n1x"]
        return far

    def CSI(self):
        """ The critical success index CSI (Schaefer, 1990) is the number of correct event forecasts divided by
            the number of cases forecast and/or observed """

        num = self.contingency
        csi = num["n11"] / (num["nxx"] - num["n00"])
        return csi

    def HSS(self):
        """ The Heidke skill score
        HSS is a measure of correct forecasts; with random correct forecasts removed (i.e. forecasts
        expected to be correct by chance). The reference forecast in HSS is random chance F, subject to
        the constraint that marginal distributions of forecasts are the same as the marginal distributions of
        observations """

        num = self.contingency
        hss = (num["n11"] * num["n00"] - num["n01"] * num["n10"]) / ((num["nx1"] * num["n0x"] + num["n1x"] * num["nx0"]) / 2)
        f = ((num["n00"] + num["n10"]) * (num["n00"] + num["n01"]) + (num["n11"] + num["n10"]) * (num["n11"] + num["n01"])) / num["nxx"]
        return hss, f


if __name__ == "__main__":

    mask = ascii.read_ascii("C:\Master\GIS/astental/mask.asc")[1]
    mask[np.isnan(mask)] = 0
    mask = mask.astype(bool)

    mask2 = resample_binary_majority(mask, (3,3))[1].astype(bool)

    dates = compare_datetime("astental", "amundsen", "practise")

    dates2 = compare_datetime("astental", "practise", "landsat")

    set1 = create_dataset("astental", model="amundsen", obs="practise", dateobject=dates, mask=mask, overwrite=True)

    set2 = create_dataset("astental", "practise", "landsat", dates2, mask2, overwrite=True)



    print("done")

