import os
import zipfile
import pandas as pd
from splinter.browser import Browser
import time
import shutil


def download(link,name):
    browser = Browser("chrome")
    browser.visit(link)
    browser.fill('username', 'SiegmannGIS')
    browser.fill('password', 'lidawu82')
    browser.find_by_id('loginButton').click()
    while not os.path.exists(name):
        time.sleep(1)
    browser.quit()

temp = "C:\Master/temp"
path = "C:\Master\landsat"
saving = "C:\Users\Marcel/Downloads"

for region in os.listdir(path):
    for folder in os.listdir(os.path.join(path, region)):
        if "availability" in folder:
            for meta in os.listdir(os.path.join(path,region,folder)):
                csv = pd.read_csv(os.path.join(path,region,folder,meta))
                for link in (csv["Download Link"]):
                    link = link.replace("/options","")+"/FR_BUND/EE"
                    # -options /STANDARD/EE
                    # -options /FR_BUND/EE
                    name = link.split("/")[-3]+".zip"
                    name2 = link.split("/")[-3] + "_QB.png"
                    print(name)
                    if not os.path.exists(os.path.join(path,region,"images",name2)):
                        download(link,os.path.join(saving,name))
                        shutil.move(os.path.join(saving,name),os.path.join(path,region,"images",name))

                        # zip it
                        with open(os.path.join(path,region,"images",name), "rb") as file:
                            zipf = zipfile.ZipFile(file)
                            for element in zipf.namelist():
                                if "_QB" in str(element):
                                    zipf.extract(element, os.path.join(path,region,"images"))

                        os.remove(os.path.join(path,region,"images",name))






