import numpy as np
import tools.ascii as ascii
import matplotlib.pyplot as plt
import math


def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return array[idx]


def angle_between(point, target, clockwise=True):
    angle = np.rad2deg(np.arctan2(target[1] - point[1], target[0] - point[0]))
    if not clockwise:
        angle = -angle
    return angle % 360


# Information
headinfo, dgm = ascii.read_ascii("C:\Master\GIS/astental/dgm.asc")
hillshade = ascii.read_ascii("C:\Master\GIS/astental/hillshade.asc")[1]
cam_x, cam_y = 373335,340327
print headinfo


