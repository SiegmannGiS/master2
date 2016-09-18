from splinter.browser import Browser
import time
import os

browser = Browser("chrome")
browser.visit("http://earthexplorer.usgs.gov/download/4923/LC81910282015278LGN00/FR_BUND/EE")
browser.fill('username', 'SiegmannGIS')
browser.fill('password', 'lidawu82')
browser.find_by_id('loginButton').click()
while not os.path.exists("C:\Users\Marcel\Downloads/LC81910282015278LGN00.zip"):
    time.sleep(1)
browser.quit()