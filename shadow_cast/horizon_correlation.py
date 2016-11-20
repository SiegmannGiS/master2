import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt
from skimage import feature
import pandas as pd
import tools.ascii as ascii
import os


def get_image_horizon(image_path, cam_chip_height, cam_chip_width, focal_length, horizontal_cutting):
    """

    :param image_path: Path to Reference Image (blue sky image without clouds)
    :param cam_chip_height: Sensor Size height [m]
    :param cam_chip_width: Sensor Size width [m]
    :param focal_length: Focal length of the camera [m]
    :param horizontal_cutting: Cut out Logos in the webcam image in the blue sky region
    :return: Pandas series Index = vertical angle [0.1 degrees], Values = horizontal angle [0.1 degrees]
    """
    # Open image
    img = Image.open(image_path)

    # Angle index
    angle_horizontal_complete = np.rad2deg(np.arctan((0.5*cam_chip_height)/focal_length)*2)
    angle_vertical_complete = np.rad2deg(np.arctan((0.5*cam_chip_width)/focal_length)*2)
    vert_lin = np.linspace(0,angle_vertical_complete, img.size[0])
    hori_lin = np.linspace(0,angle_horizontal_complete, img.size[1])
    xv, yv = np.meshgrid(vert_lin, hori_lin)
    yv = np.flipud(yv)

    # Edge detection
    img_rast = np.array(img.convert(mode="L"))
    edges = feature.canny(img_rast, sigma=1)

    # Edges as line
    edges[:horizontal_cutting,:] = False
    image_skyline = np.argmax(edges == True, axis=0)
    image_skyline_x = vert_lin
    image_skyline_y = yv[image_skyline,np.arange(len(vert_lin))]

    # Interpolate to steps
    complete_series = pd.Series(image_skyline_y, image_skyline_x, name="image_horizon")
    new_index = np.sort(np.concatenate((complete_series.index, np.arange(0,angle_vertical_complete,0.1))))
    complete_series = complete_series.reindex(new_index)
    step_series = complete_series.interpolate(method="index")
    step_series = step_series[np.isnan(complete_series)]
    step_inclination = step_series.shift(1)-step_series

    plt.figure(5)
    plt.imshow(img)
    plt.plot(np.arange(img_rast.shape[1]), image_skyline, "r-")

    return step_series, step_inclination


def get_raster_skyline(path, raster_path, cam_position_x, cam_position_y, camera_offset):
    """

    :param raster_path: Path to Digital Elevation Model (ESRI ASCII Grid)
    :param cam_position_x: X Coordinate in same projection like DEM [m]
    :param cam_position_y: Y Coordinate in same projection like DEM [m]
    :param camera_offset: Camera Offset from Ground [m]
    :return: Pandas series Index = vertical angle [0.1 degrees], Values = horizontal angle [0.1 degrees]
    """

    def find_nearest(array, value):
        idx = (np.abs(array - value)).argmin()
        return array[idx]

    def angle_between(point, target, clockwise=True):
        angle = np.rad2deg(np.arctan2(target[1] - point[1], target[0] - point[0]))
        if not clockwise:
            angle = -angle
        return angle % 360

    # Information
    headinfo, dgm = ascii.read_ascii(raster_path)


    # Create Coordinate Ranges
    x_range = np.arange(headinfo[2], headinfo[2] + headinfo[0] * headinfo[4], headinfo[4])
    y_range = np.arange(headinfo[3] + headinfo[1] * headinfo[4], headinfo[3], -headinfo[4])

    # Coordinate meshgrid
    xv_coord, yv_coord = np.meshgrid(x_range, y_range)

    # Index meshgrid
    xv_index, yv_index = np.meshgrid(np.arange(dgm.shape[1]), np.arange(dgm.shape[0]))

    # find index of cam_coordinates
    x_coord, y_coord = find_nearest(x_range, cam_position_x), find_nearest(y_range, cam_position_y)
    y0, x0 = np.where((xv_coord == x_coord) & (yv_coord == y_coord))

    # find outer bound pixels of raster (index)
    x_bound = np.concatenate(
        [xv_index[y0:, -1], np.flipud(xv_index[-1, :-1]), xv_index[:-1, 0], xv_index[0, 1:], xv_index[1:y0, -1]])
    y_bound = np.concatenate(
        [yv_index[y0:, -1], yv_index[-1, :-1], np.flipud(yv_index[:-1, 0]), yv_index[0, 1:], yv_index[1:y0, -1]])

    # x_bound = np.concatenate([xv_index[0,:],xv_index[:,-1]])
    # y_bound = np.concatenate([yv_index[0,:],xv_index[:,-1]])

    point_pairs = zip(x_bound, y_bound)

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
        length = int(np.hypot(x1 - x0, y1 - y0))
        x, y = np.linspace(x0, x1, length), np.linspace(y0, y1, length)

        # Extract the values along the line
        y_int = y.astype(np.int)
        x_int = x.astype(np.int)
        x_int[x_int >= headinfo[0]] = headinfo[0] - 1
        y_int[y_int >= headinfo[1]] = headinfo[1] - 1
        zi = dgm[y_int, x_int]

        # calculate important components
        dist = np.sqrt(np.power((x_range[x0] - x_range[x_int]), 2) + np.power((y_range[y0] - y_range[y_int]), 2))
        height_difference = zi - (dgm[y0, x0] + camera_offset)
        height_angle = np.rad2deg(np.arctan(height_difference / dist))
        direction = angle_between((x0, y0), (x1, y1))
        max_angle = np.nanmax(height_angle)

        # Skyline Position in raster
        x_pos, y_pos = x[height_angle == max_angle].astype(np.int), y[height_angle == max_angle].astype(np.int)
        if len(x_pos) > 1:
            x_pos = np.array(x_pos[0])
        if len(y_pos) > 1:
            y_pos = np.array(y_pos[0])
        if not (x_pos, y_pos) == point:
            all_x_pos.append(x_pos)
            all_y_pos.append(y_pos)
            all_height.append(dgm[y_pos, x_pos])
            all_direction.append(direction)
            all_max_angle.append(max_angle)
            dist_max = dist[height_angle == max_angle]
            if len(dist_max) > 1:
                dist_max = np.array(dist_max[0])
            all_dist.append(dist_max)

    # Pandas Series Interpolate to 0.1 degree steps
    complete_series = pd.Series(np.array(all_max_angle), np.array([x[0] for x in all_direction]),
                                name="view_angles_360")
    new_index = np.sort(np.concatenate((complete_series.index, np.arange(0, 360.1, 0.1))))
    complete_series = complete_series.reindex(new_index)
    step_series = complete_series.interpolate(method="index")
    step_series360 = step_series[np.isnan(complete_series)]
    step_series720 = pd.Series(np.concatenate([step_series360.values,step_series360.values]), np.concatenate([step_series360.index,step_series360.index+360]))
    step_inclination720 = step_series720.shift(1)-step_series720

    # Plots
    plt.figure(1)
    plt.polar(np.deg2rad(all_direction), all_dist, 'r-')

    plt.figure(2)
    plt.polar(np.deg2rad(all_direction), all_max_angle, 'b-')

    # plt.figure(3)
    # hillshade = ascii.read_ascii("C:\Master\GIS/vernagtferner/hillshade_vernagtferner_5m.asc")[1]
    skyline = np.zeros_like(dgm)
    skyline[all_y_pos, all_x_pos] = dgm[all_y_pos, all_x_pos]
    # plt.imshow(hillshade, cmap="gray")
    # plt.plot(all_x_pos,all_y_pos, "r-")

    ascii.write_ascii(os.path.join(path,"skyline.asc"), headinfo, skyline, format="%f")

    return step_series360, step_series720, step_inclination720


def RMSE_horizon(raster, image, step_series_image):
    """
    Calculates the RMSE between the Image Horizon and the Raster
    :param step_inclination_raster_series: Pandas Series step_inclination_ from function get_raster_skyline
    :param step_inclination_image_series: Pandas Series step_inclination_ from function get_image_horizon
    :return: lowest correlation
    """

    raster_values = raster.values[1:]
    image_values = image.values[1:]

    rmse = []
    for i, num in enumerate(raster_values):
        if i <= len(raster_values) - len(image_values):
            compare = np.array(raster_values[i:i + len(image_values)])
            # result.append(comp.correlation(compare,match))
            rmse.append(((compare - image_values) ** 2).mean())

    index = np.where(rmse == np.min(rmse))[0]
    if len(index) > 1:
        index = index[0]

    raster_position_image = pd.Series(step_series_image.values, image.index + raster.index[index])
    raster_correlation_image = pd.Series(image.values, image.index + raster.index[index])

    return raster_position_image, raster_correlation_image


if __name__ == "__main__":
    # Image
    cam_hei= 0.0147
    cam_wid= 0.0222
    cam_foc=0.02
    image_path = "C:\Master\settings/vernagtferner/k2015-08-03_1132_VKA_6822.JPG"
    path = "C:\Master\GIS/vernagtferner"

    step_series_image, step_inclination_image = get_image_horizon(image_path,cam_hei,cam_wid,cam_foc, 300)

    # Raster
    rast = "C:\Master\GIS/vernagtferner/dgm5_vernagtferner.asc"
    cam_x, cam_y = 209389.7,332820.2
    offset = 3

    step_series360_raster, step_series720_raster, step_inclination720_raster = get_raster_skyline(path, rast, cam_x, cam_y, offset)

    # # RMSE
    # raster_position_image, raster_correlation_image = RMSE_horizon(step_inclination720_raster,step_inclination_image,step_series_image)
    # min_rast_area = np.nanmin(step_series720_raster[raster_position_image.index].values)
    # min_image = np.nanmin(raster_position_image.values)
    #
    # raster_position_image = raster_position_image + (min_rast_area - min_image)
    #
    # # ORB TEst
    # # xv_coord, yv_coord = np.meshgrid(raster_position_image.index, raster_position_image.values)
    # # empty_image = np.zeros_like(xv_coord)
    # # empty_image[:] = 255
    # # image[raster_position_image.values,]
    #
    # print("done")
    #
    # ## Plot Correlation
    # plt.figure(3)
    # plt.plot(step_series720_raster)
    # plt.plot(raster_position_image)
    #
    # plt.figure(4)
    # plt.plot(step_inclination720_raster)
    # plt.plot(raster_correlation_image)
    #
    # plt.show()