import lensfunpy
import cv2

camMaker = 'Canon'
camModel = 'Canon EOS 1100D'
lensMaker = 'Canon'
lensModel = 'Canon EF-S 18-55mm f/3.5-5.6 IS II'

db = lensfunpy.Database()
cam = db.find_cameras(camMaker, camModel)[0]
lens = db.find_lenses(cam, lensMaker, lensModel)[0]
print(cam)
print(lens)


focal_length = 20.0
aperture = 3.5
distance = 1500
image_path = "2014-10-28_09-00.jpg"
undistorted_image_path = "2014-10-28_09-00_dist.jpg"

im = cv2.imread(image_path)
height, width = im.shape[0], im.shape[1]

mod = lensfunpy.Modifier(lens, cam.crop_factor, width, height)
mod.initialize(focal_length, aperture, distance)

undist_coords = mod.apply_geometry_distortion()
im_undistorted = cv2.remap(im, undist_coords, None, cv2.INTER_LANCZOS4)
cv2.imwrite(undistorted_image_path, im_undistorted)