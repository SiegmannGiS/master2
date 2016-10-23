import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt
from skimage import feature
import pandas as pd


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
    edges = feature.canny(img_rast, sigma=2)

    # Edges as line
    edges[:horizontal_cutting,:] = False
    image_skyline = np.argmax(edges.nonzero(), axis=0)
    image_skyline_x = vert_lin
    image_skyline_y = yv[image_skyline,np.arange(len(vert_lin))]

    # Interpolate to steps
    complete_series = pd.Series(image_skyline_y, image_skyline_x, name="image_horizon")
    new_index = np.sort(np.concatenate((complete_series.index, np.arange(0,angle_vertical_complete,0.1))))
    complete_series = complete_series.reindex(new_index)
    step_series = complete_series.interpolate(method="index")
    step_series = step_series[np.isnan(complete_series)]

    return step_series

cam_hei= 0.0147
cam_wid= 0.0222
cam_foc= 0.02
image_path = "C:\Master\settings/astental/ref-2016_08-24-1500.jpg"

step_series = get_image_horizon(image_path,cam_hei,cam_wid,cam_foc, 300)


