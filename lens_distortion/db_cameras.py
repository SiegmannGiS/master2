import lensfunpy
import cv2
import os
from skimage.io import imread
import numpy as np


class LensDistortion(object):
    def __init__(self):
        self.cameradict = {"astental": {"camMaker": 'Canon', "camModel": 'Canon EOS 1100D', "lensMaker": 'Canon',
                                   "lensModel": 'Canon EF-S 18-55mm f/3.5-5.6 IS II'},
                      "vernagtferner": {"camMaker": 'Nikon Corporation', "camModel": 'Nikon D200', "lensMaker": 'Tokina',
                                        "lensModel": 'Tokina AF 12-24mm f/4 AT-X Pro DX'}}
        self.db = lensfunpy.Database()

    def addCamParameter(self, site, camMaker, camModel, lensMaker, lensModel):
        self.cameradict[site] = {"camMaker": camMaker, "camModel": camModel, "lensMaker": lensMaker,
                                   "lensModel": lensModel}

    def checkCamSetup(self,site):
        cam = self.db.find_cameras(self.cameradict[site]["camMaker"], self.cameradict[site]["camModel"])[0]
        lens = self.db.find_lenses(cam, self.cameradict[site]["lensMaker"], self.cameradict[site]["lensModel"])[0]
        if len(cam) == 0:
            print("Camera not found")
        if len(lens) == 0:
            print("Lens not found")

        if not len(cam) == 0 or not len(lens) == 0:
            print(cam, lens)

    def correctImage(self, site, imagepath, correctedpath, focal_length, aperture, distance):

        cam = self.db.find_cameras(self.cameradict[site]["camMaker"], self.cameradict[site]["camModel"])[0]
        lens = self.db.find_lenses(cam, self.cameradict[site]["lensMaker"], self.cameradict[site]["lensModel"])[0]

        im = cv2.imread(imagepath)
        height, width = im.shape[0], im.shape[1]

        mod = lensfunpy.Modifier(lens, cam.crop_factor, width, height)
        mod.initialize(focal_length, aperture, distance)

        undist_coords = mod.apply_geometry_distortion()
        im_undistorted = cv2.remap(im, undist_coords, None, cv2.INTER_LANCZOS4)
        cv2.imwrite(correctedpath, im_undistorted)


if __name__ == "__main__":
    path = "C:\Master\settings"
    path_new = "C:\Master\settings"
    site = "astental"
    ld = LensDistortion()

    # for element in os.listdir(os.path.join(path,site)):
    #     print(element)
    #     ld.correctImage(site, os.path.join(path,site,element), os.path.join(path_new,site,element), 12.0, 4.0, 700)

    # ld.correctImage(site, os.path.join(path, site, "2015-08-03_1132_VKA_6822.JPG"), os.path.join(path_new, site, "2015-08-03_1132_corrected.JPG"), 12.0, 4.0, 700)

    mask = cv2.imread("2015-04-14_09-00.jpg").transpose()

    mask[:, ::200, :] = 0
    mask[:, 1::200, :] = 0
    mask[:, 2::200, :] = 0
    mask[:, 3::200, :] = 0
    mask[:, 4::200, :] = 0
    mask[:, :, ::200] = 0
    mask[:, :, 1::200] = 0
    mask[:, :, 2::200] = 0
    mask[:, :, 3::200] = 0
    mask[:, :, 4::200] = 0

    mask = mask.transpose()
    cv2.imwrite("gitter.jpg", mask)


    ld.correctImage("astental", "gitter.jpg", "2015-04-14_09-00_gitter.jpg", 20.0, 3.5, 1500)

    mask = cv2.imread("2015-04-14_09-00_gitter.jpg").transpose()

    mask[2, ::200, :] = 255
    mask[2, 1::200, :] = 255
    mask[2, 2::200, :] = 255
    mask[2, 3::200, :] = 255
    mask[2, 4::200, :] = 255
    mask[2, :, ::200] = 255
    mask[2, :, 1::200] = 255
    mask[2, :, 2::200] = 255
    mask[2, :, 3::200] = 255
    mask[2, :, 4::200] = 255
    mask = mask.transpose()

    cv2.imwrite("2015-04-14_09-00_gitter_neu.jpg", mask)

