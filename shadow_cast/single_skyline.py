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

# Create Coordinate meshgrid
x_range = np.arange(headinfo[2],headinfo[2]+headinfo[0]*headinfo[4],headinfo[4])
y_range = np.arange(headinfo[3]+headinfo[1]*headinfo[4],headinfo[3], -headinfo[4])

xv, yv = np.meshgrid(x_range,y_range)

# find index of cam_coordinates
x_coord, y_coord = find_nearest(x_range, cam_x), find_nearest(y_range, cam_y)
y0, x0 = np.where((xv == x_coord) & (yv == y_coord))

# index_meshgrid
x_bound = np.concatenate([np.arange(int(headinfo[0])), np.full(headinfo[1]-2, headinfo[0]-1), np.flipud(np.arange(int(headinfo[0]))), np.full(int(headinfo[1]-2), 0)])

# Algorithm
x1, y1 = 0,y0
length = int(np.hypot(x1-x0, y1-y0))
x, y = np.linspace(x0, x1, length), np.linspace(y0, y1, length)

# Extract the values along the line
zi = dgm[y.astype(np.int), x.astype(np.int)]

# calculate important components
dist = np.sqrt(np.power((x_range[x0] - x_range[x.astype(np.int)]), 2) + np.power((y_range[y0] - y_range[y.astype(np.int)]), 2))
height_difference = zi - dgm[y0, x0]
height_angle = np.rad2deg(np.arctan(height_difference / dist))
direction = angle_between((x0, y0), (x1, y1))
max_angle = np.nanmax(height_angle)

# Skyline Position in raster
x_pos, y_pos = x[height_angle == max_angle].astype(np.int), y[height_angle == max_angle].astype(np.int)

print "done"

fig, axes = plt.subplots(nrows=2)
axes[0].imshow(hillshade, cmap="gray")
axes[0].plot([x0, x1], [y0, y1], 'ro-')
axes[0].plot(x_pos, y_pos, 'bo')
axes[0].axis('image')

axes[1].plot(dist,zi)
axes[1].plot([0, dist[height_angle == max_angle]], [dgm[y0, x0], dgm[y_pos,x_pos]], 'r-')
axes[1].plot(0, dgm[y0, x0], 'ro')
axes[1].plot(dist[height_angle == max_angle], dgm[y_pos,x_pos], 'bo')
plt.show()
