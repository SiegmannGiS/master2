import os
import datetime
import urllib2, urllib
from bs4 import BeautifulSoup
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from skimage.color import rgb2hsv
from skimage.io import imread
from calendar import monthrange
import time
from skimage import feature


def percentage(part, whole):
    return 100 * float(part)/float(whole)


def roundTime(dt=None, dateDelta=datetime.timedelta(minutes=1)):
    """Round a datetime object to a multiple of a timedelta
    dt : datetime.datetime object, default now.
    dateDelta : timedelta object, we round to a multiple of this, default 1 minute.
    Author: Thierry Husson 2012 - Use it as you want but don't blame me.
            Stijn Nevens 2014 - Changed to use only datetime objects as variables
    """
    roundTo = dateDelta.total_seconds()

    if dt == None : dt = datetime.datetime.now()
    seconds = (dt - dt.min).seconds
    # // is a floor division, not a comment on following line:
    rounding = (seconds+roundTo/2) // roundTo * roundTo
    return dt + datetime.timedelta(0,rounding-seconds,-dt.microsecond)


def read_links(line):
    value = line.split(",")
    value[5] = [int(x) for x in value[5][1:-1].split(" ")]
    key = ["name", "typ", "link", "startmonth", "endmonth", "years", "imagesperday"]
    info_dict = dict(zip(key, value))
    if info_dict["typ"] == "fotowebcam":
        info_dict["imagesperday"] = info_dict["imagesperday"][1:-1].split(" ")
    return info_dict


def vernagtferner_getimages(info_dict):

    def vernagtferner_getimages2(downloadlink, year, yearbol=False):
        html = BeautifulSoup(urllib2.urlopen(downloadlink).read(), "lxml")
        imagelinks = []
        for a in html.find_all('a', href=True):
            if "bilder" in a["href"]:
                if yearbol == True:
                    imagelink = "http://vernagt.userweb.mwn.de/vkam_archiv/%s/%s" % (str(year), a["href"])
                else:
                    imagelink = "http://vernagt.userweb.mwn.de/vkam_archiv/" + a["href"]

                date = datetime.datetime.strptime(
                    str(a["href"])[str(a["href"]).index("k") + 1:str(a["href"]).index("k") + 16],
                    "%Y-%m-%d_%H%M")
                settime = str(a)[str(a).index("alt") + 5:str(a).index("alt") + 11].split(":")
                settime = datetime.time(hour=int(settime[0]), minute=int(settime[1]))
                imagedict = {"settime": settime, "date": date}
                imagelinks.append((imagelink, imagedict))
        return imagelinks

    imagelinks = []
    for i,year in enumerate(info_dict["years"]):
        if year == datetime.datetime.today().year:
            downloadlink = info_dict["link"]
            imagelinks += vernagtferner_getimages2(downloadlink, year)
        else:
            downloadlink = info_dict["link"].split("/")
            downloadlink.insert(4,str(year))
            downloadlink[5] = downloadlink[5].split(".")
            downloadlink[5].insert(1,str(year))
            downloadlink[5] = "".join(downloadlink[5])
            downloadlink[5] = downloadlink[5][:-4]+"."+downloadlink[5][-4:]
            downloadlink = "/".join(downloadlink)
            imagelinks += vernagtferner_getimages2(downloadlink,year,yearbol=True)

    return imagelinks


def fotowebcam_getimages(info_dict):

    def createlink(start,end):
        liste = []
        for month in range(start,end+1):
            for day in range(1, monthrange(year, month)[1]+1):
                for time in info_dict["imagesperday"]:
                    date = datetime.datetime.strftime(datetime.datetime(year, month, day), "%Y/%m/%d")
                    liste.append("%s/%s/%s_la.jpg" % (info_dict["link"], date, time))
        return liste

    downloadlinks = []

    for i, year in enumerate(info_dict["years"]):
        if i == 0:
            downloadlinks.append(createlink(int(info_dict["startmonth"]), 12))

        elif i == len(info_dict["years"])-1:
            downloadlinks.append(createlink(1, int(info_dict["endmonth"])))

        else:
            downloadlinks.append(createlink(1, int(info_dict["endmonth"])))
            downloadlinks.append(createlink(int(info_dict["startmonth"]), 12))

    downloadlinks = [item for sublist in downloadlinks for item in sublist]

    return downloadlinks


def reduce_links_by_time(imagelinks, info_dict):
    imagelinks_true = []
    for link in imagelinks:

        if link[1]["date"].year == info_dict["years"][0]:
            if link[1]["date"].month >= int(info_dict["startmonth"]):
                imagelinks_true.append(link)
        else:
            if link[1]["date"].month >= int(info_dict["startmonth"]):
                imagelinks_true.append(link)
            elif link[1]["date"].month <= int(info_dict["endmonth"]):
                imagelinks_true.append(link)

    return imagelinks_true


def download_vernagt(info_dict,imagelinks,mainpath):

    def edge(pathfile):
        image = imread(pathfile, as_grey=True)
        image = image[190:381, :]
        edges1 = feature.canny(image, sigma=2)
        num = np.sum(edges1)
        return num

    if not os.path.exists(os.path.join(mainpath,"images",info_dict["name"])):
        os.makedirs(os.path.join(mainpath,"images",info_dict["name"]))
    temppath = os.path.join(mainpath,"temp","temp.jpg")

    refimagepath = "C:\Master\settings/vernagtferner14-16/ref_2016-03-26_1125_VKA_7414.JPG"
    refnum = edge(refimagepath)

    for imagelink in imagelinks:
        print "Work on: "+str(imagelink[0])

        #roundtime = roundTime(imagelink[1]["date"], datetime.timedelta(minutes=30))
        roundtime = imagelink[1]["date"]
        imagename = datetime.datetime.strftime(roundtime, "%Y-%m-%d_%H-%M_0.jpg")
        imagepath = os.path.join(mainpath, "images", info_dict["name"], imagename)

        if not os.path.exists(imagepath):
            urllib.urlretrieve(imagelink[0], temppath)

            num = edge(temppath)
            prozent = num/refnum*100
            if prozent >= 90:
                print "clear image"
                os.rename(temppath,imagepath)





def download_fotowebcam(info_dict,imagelinks,mainpath):
    if not os.path.exists(os.path.join(mainpath, "images", info_dict["name"])):
        os.makedirs(os.path.join(mainpath, "images", info_dict["name"]))
    temppath = os.path.join(mainpath, "temp", "temp.jpg")

    with np.load("%s/settings/%s/settings.npz" % (mainpath, info_dict["name"])) as settings:
        clearmask = settings["clearmask"]
        skymask = settings["skymask"]

    downloadamount = 0
    if downloadamount >= 400000000:
        time.sleep(1800)
        print "Wait half an hour"

    for imagelink in imagelinks:
        print "Work on: " + str(imagelink)
        urllib.urlretrieve(imagelink, temppath)
        downloadamount += os.path.getsize(temppath)

        if not os.path.getsize(temppath) < 500:
            img = imread(temppath)
            hsv = rgb2hsv(img)

            clearchannel = (hsv[:, :, 2] * 100).astype(int)
            cleardecision = clearchannel[clearmask] < 50

            if np.sum(cleardecision) >= 3:
                skychannel = (hsv[:, :, 1] * 100).astype(int)
                skydecision = skychannel[skymask] < 30
                cloud = int(percentage(np.sum(skydecision), np.sum(skymask)))
                imagename = imagelink.split("/")[-4:]
                imagename = "%s_%s-%s_%s.jpg" % ("-".join(imagename[0:3]), imagename[3][:2], imagename[3][2:4], cloud)
                imagepath = os.path.join(mainpath,"images",info_dict["name"],imagename)
                imagelink = imagelink.replace("_la.","_hu.")
                print "Clear image: " + imagename
                if not os.path.exists(imagepath):
                    urllib.urlretrieve(imagelink, imagepath)



if __name__ == "__main__":
    outpath = "C:/master"

    with open(os.path.join(outpath,"settings/downloadlinks.txt")) as fobj:
        lines = fobj.readlines()

    for i,line in enumerate(lines):
        if i != 0:
            if not line[0] == "#":
                info_dict = read_links(line)

                if info_dict["typ"] == "vernagtferner":
                    # Get imagelinks
                    imagelinks = vernagtferner_getimages(info_dict)

                    # Choose by given time information
                    imagelinks = reduce_links_by_time(imagelinks,info_dict)

                    # Download and check weather situation
                    download_vernagt(info_dict,imagelinks,outpath)

                if info_dict["typ"] == "fotowebcam":
                    # Get imagelinks
                    imagelinks = fotowebcam_getimages(info_dict)

                    # Choose by given time information
                    download_fotowebcam(info_dict, imagelinks, outpath)