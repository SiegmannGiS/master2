import numpy as np


cam_chip_height = 0.0147
cam_chip_width = 0.0222
focal_length = 0.02
img_width = 4272
img_height = 2848
distance = 3500

angle_horizontal_complete = np.rad2deg(np.arctan((0.5*cam_chip_height)/focal_length)*2)
angle_vertical_complete = np.rad2deg(np.arctan((0.5*cam_chip_width)/focal_length)*2)

one_pix_deg_height = 90 - angle_horizontal_complete/float(img_height)
one_pix_deg_width = 90 - angle_vertical_complete/float(img_width)



print(one_pix_deg_width)

width = 1/np.tan(np.deg2rad(one_pix_deg_width)) * distance
height = 1/np.tan(np.deg2rad(one_pix_deg_height)) * distance

print(width, height)