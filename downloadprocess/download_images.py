import os, sys
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
from pytz import timezone
from shutil import move
import logging


def percentage(part, whole):
    return 100 * float(part)/float(whole)


def read_links(line):
    value = line.split(",")
    value[5] = [int(x) for x in value[5][1:-1].split(" ")]
    key = ["name", "typ", "link", "startmonth", "endmonth", "years", "imagesperday", "timezone", "daylight_saving"]
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
                    dobject = datetime.datetime(year, month, day, int(time[:2]), int(time[2:]))
                    date = timezone("UTC").localize(dobject)
                    date = date.astimezone(timezone(info_dict["timezone"]))
                    date = datetime.datetime.strftime(date, "%Y/%m/%d/%H%M_la.jpg")
                    liste.append("%s/%s" % (info_dict["link"], date))

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

    refimagepath = "C:\Master\settings/%s/ref_2016-03-26_1125_VKA_7414.JPG" % info_dict["name"]
    refnum = edge(refimagepath)

    for imagelink in imagelinks:
        print "Work on: "+str(imagelink[0])

        # Get Image date
        date = imagelink[1]["date"]
        date = timezone("UTC").localize(date - datetime.timedelta(hours=1))


        imagename = date.strftime("%Y-%m-%d_%H-%M.jpg")
        imagepath = os.path.join(mainpath, "images", info_dict["name"], imagename)
        print(imagepath)

        if not os.path.exists(imagepath):
            print(imagepath)
            urllib.urlretrieve(imagelink[0], temppath)
            print("download")

            num = edge(temppath)
            prozent = num/refnum*100
            if prozent >= 90:
                print "clear image"
                os.rename(temppath,imagepath)


def download_fotowebcam(info_dict,imagelinks,mainpath):

    def edge(pathfile,maskpath):
        image = imread(pathfile, as_grey=True)
        mask = imread(maskpath, as_grey=True)
        mask = mask.astype(bool)
        edges = feature.canny(image, sigma=2)
        num = np.sum(edges[mask])
        return float(num),edges

    if not os.path.exists(os.path.join(mainpath, "images", info_dict["name"])):
        os.makedirs(os.path.join(mainpath, "images", info_dict["name"]))
    temppath = os.path.join(mainpath, "temp", "temp.jpg")

    downloadamount = 0
    if downloadamount >= 400000000:
        time.sleep(1800)
        print("Wait half an hour")

    for i, imagelink in enumerate(imagelinks):


        # Get Image date
        date = datetime.datetime.strptime("-".join(imagelink.split("/")[-4:]), "%Y-%m-%d-%H%M_la.jpg")
        if info_dict["daylight_saving"]:
            date = timezone(info_dict["timezone"]).localize(date)
            date = date.astimezone(timezone('UTC'))
        else:
            date = timezone("UTC").localize(date - datetime.timedelta(hours=1))

        img_name = date.strftime("%Y-%m-%d_%H-%M.jpg")
        img_path = os.path.join(mainpath, "images", info_dict["name"], img_name)

        if not os.path.exists(img_path):
            print("Work on: " + str(imagelink))
            urllib.urlretrieve(imagelink, temppath)
            downloadamount += os.path.getsize(temppath)

            if not os.path.getsize(temppath) < 500:

                # Calculate Edge Pixel Percentage
                maskpath = os.path.join(mainpath, "settings/%s/ref_la.jpg" % info_dict["name"])
                if not os.path.exists(os.path.join(mainpath, "temp","last_one.jpg")):
                    refimagepath = os.path.join(mainpath, "settings/%s/ref_la.jpg" % info_dict["name"])
                    refnum, edges = edge(refimagepath, maskpath)
                else:
                    refimagepath = os.path.join(mainpath, "temp","last_one.jpg")
                    refnum, edges = edge(refimagepath, maskpath)


                num, edges = edge(temppath, maskpath)
                prozent = num / refnum * 100
                print(prozent, " %")

                # Check Weather condition
                if prozent >= 0:


                    # Create Name
                    imagelink = imagelink.replace("_la.", "_hu.")
                    print("Clear image: " + img_name)
                    logging.info("%s: %s percent" % (img_name,prozent))

                    # Last image _la
                    move(temppath, os.path.join(mainpath, "temp","last_one.jpg"))

                    # Download real Image
                    urllib.urlretrieve(imagelink, img_path)


if __name__ == "__main__":
    logging.basicConfig(filename='image_info.log', level=logging.DEBUG)

    outpath = "C:/master"

    with open(os.path.join(outpath,"settings/downloadlinks.txt")) as fobj:
        lines = fobj.readlines()

    for i,line in enumerate(lines):
        if i != 0:
            if not line[0] == "#":
                info_dict = read_links(line)

                print info_dict

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
                    print(imagelinks)

                    # Choose by given time information
                    download_fotowebcam(info_dict, imagelinks, outpath)