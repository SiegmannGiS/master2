import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import urllib
import os

test = urllib.urlretrieve("http://www.foto-webcam.eu/webcam/wallackhaus-nord/2016/04/30/1630_hu.jpg", "test.jpg")
print type(os.path.getsize("test.jpg"))

test2 = urllib.urlretrieve("http://www.foto-webcam.eu/webcam/wallackhaus-nord/2016/03/15/0830_la.jpg","test2.jpg")
print os.path.getsize("test2.jpg")
print test
print test2