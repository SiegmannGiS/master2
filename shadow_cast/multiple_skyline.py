import numpy as np
import tools.ascii as ascii
import pandas as pd
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
headinfo, dgm = ascii.read_ascii("C:\Master\GIS/astental/dgm_big.asc")
hillshade = ascii.read_ascii("C:\Master\GIS/astental/hillshade_big.asc")[1]
cam_x, cam_y = 373335,340327
offset = 5

# Create Coordinate Ranges
x_range = np.arange(headinfo[2],headinfo[2]+headinfo[0]*headinfo[4],headinfo[4])
y_range = np.arange(headinfo[3]+headinfo[1]*headinfo[4],headinfo[3], -headinfo[4])

# Coordinate meshgrid
xv_coord, yv_coord = np.meshgrid(x_range,y_range)

# Index meshgrid
xv_index, yv_index = np.meshgrid(np.arange(dgm.shape[1]), np.arange(dgm.shape[0]))

# find index of cam_coordinates
x_coord, y_coord = find_nearest(x_range, cam_x), find_nearest(y_range, cam_y)
y0, x0 = np.where((xv_coord == x_coord) & (yv_coord == y_coord))

# find outer bound pixels of raster (index)
x_bound = np.concatenate([xv_index[y0:, -1], np.flipud(xv_index[-1, :-1]), xv_index[:-1, 0], xv_index[0, 1:], xv_index[1:y0, -1]])
y_bound = np.concatenate([yv_index[y0:, -1], yv_index[-1, :-1], np.flipud(yv_index[:-1, 0]), yv_index[0, 1:], yv_index[1:y0, -1]])

# x_bound = np.concatenate([xv_index[0,:],xv_index[:,-1]])
# y_bound = np.concatenate([yv_index[0,:],xv_index[:,-1]])

point_pairs = zip(x_bound,y_bound)

## Iterate over bounding box pixels
all_direction = []
all_height = []
all_max_angle = []
all_x_pos = []
all_y_pos = []
all_dist = []

for point in point_pairs:

    ## Algorithm
    x1, y1 = point[0], point[1]
    length = int(np.hypot(x1-x0, y1-y0))
    x, y = np.linspace(x0, x1, length), np.linspace(y0, y1, length)

    # Extract the values along the line
    y_int = y.astype(np.int)
    x_int = x.astype(np.int)
    x_int[x_int >= headinfo[0]] = headinfo[0] - 1
    y_int[y_int >= headinfo[1]] = headinfo[1] - 1
    zi = dgm[y_int, x_int]

    # calculate important components
    dist = np.sqrt(np.power((x_range[x0] - x_range[x_int]), 2) + np.power((y_range[y0] - y_range[y_int]), 2))
    height_difference = zi - (dgm[y0, x0]+offset)
    height_angle = np.rad2deg(np.arctan(height_difference / dist))
    direction = angle_between((x0, y0), (x1, y1))
    max_angle = np.nanmax(height_angle)

    # Skyline Position in raster
    x_pos, y_pos = x[height_angle == max_angle].astype(np.int), y[height_angle == max_angle].astype(np.int)
    if len(x_pos) > 1:
        x_pos = np.array(x_pos[0])
    if len(y_pos) > 1:
        y_pos = np.array(y_pos[0])
    if not (x_pos,y_pos) == point:
        all_x_pos.append(x_pos)
        all_y_pos.append(y_pos)
        all_height.append(dgm[y_pos, x_pos])
        all_direction.append(direction)
        all_max_angle.append(max_angle)
        dist_max = dist[height_angle == max_angle]
        if len(dist_max) > 1:
            dist_max = np.array(dist_max[0])
        all_dist.append(dist_max)
    # else:
    #     all_x_pos.append(np.nan)
    #     all_y_pos.append(np.nan)
    #     all_height.append(np.nan)
    #     all_direction.append(direction)
    #     all_max_angle.append(np.nan)

# fig, axes = plt.subplots(nrows=2)
# axes[0].imshow(hillshade, cmap="gray")
# axes[0].plot(x0, y0, 'ro')
# axes[0].plot(all_x_pos, all_y_pos, 'b-')
# axes[0].axis('image')
#
# axes[1].plot(all_direction,all_max_angle)
# plt.figure(1)
# plt.plot(all_direction,all_max_angle)
#
# plt.figure(2)
# ax = plt.subplot(1, 1, 1, projection="polar")
# # for x,y in zip(all_direction,all_dist):
# #     plt.polar(np.deg2rad(x), y, 'ro')
# plt.polar(np.deg2rad(all_direction), all_height, 'r-')
# plt.show()


# Pandas tests
complete_series = pd.Series(np.array(all_max_angle), np.array([x[0] for x in all_direction]), name="view_angles_360")
new_index = np.sort(np.concatenate((complete_series.index, np.arange(0,360.1,0.1))))
complete_series = complete_series.reindex(new_index)
step_series = complete_series.interpolate(method="index")
step_series = step_series[np.isnan(complete_series)]
# step_series = pd.rolling_mean(step_series, window=5, center = True)

# plt.plot(step_series)
# plt.show()

print "done"