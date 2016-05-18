# -*- coding: utf-8 -*-
from osgeo import gdal
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse


def raster2array(rastername):
    raster = gdal.Open(rastername)
    band = raster.GetRasterBand(1)
    array = band.ReadAsArray()
    array = array.astype(float)
    return array

def gdalout(inrast,outrastname,outarray,outformat):

    #Dimension des Rasters bzw. Arrays
    x_pix = inrast.RasterXSize
    y_pix = inrast.RasterYSize

    #origin and pixel size taken from input data band1
    geotransform = inrast.GetGeoTransform()
    #print geotransform

    #pixel size in east-west and north-south direction
    psize_we = geotransform[1]
    psize_ns = geotransform[5]

    #origin: x_min (top left corner),x_max (top left corner)
    x_min = geotransform[0]
    y_max = geotransform[3]

    #projection from inrast
    wkt_projection = inrast.GetProjection()

    #set driver for output format
    driver = gdal.GetDriverByName(outformat)

    #create outfile
    dataset = driver.Create(outrastname, x_pix, y_pix, 1, gdal.GDT_Float32, )
    dataset.SetGeoTransform((x_min,psize_we,0,y_max,0,psize_ns))
    dataset.SetProjection(wkt_projection)
    dataset.GetRasterBand(1).WriteArray(outarray)
    dataset.FlushCache()

    return True

#path
path = "C:\Master\landsat/vernagtferner14-16"

date = "2015-04-10_10-03"

# Bands
imgb6 = "B6_toar.tif"
imgb3 = "B3_toar.tif"

b6 = raster2array(os.path.join(path,imgb6))
b3 = raster2array(os.path.join(path,imgb3))
NDSI = (b3-b6)/(b3+b6)

low_ndsi = np.zeros_like(NDSI)
low_ndsi[NDSI >= 0.4] = 1

medium_ndsi = np.zeros_like(NDSI)
medium_ndsi[NDSI >= 0.5] = 1

high_ndsi = np.zeros_like(NDSI)
high_ndsi[NDSI > 0.7] = 1

inrast = gdal.Open(os.path.join(path,imgb6))

#gdalout(inrast,os.path.join(path,"low_ndsi_%s.tif" %date),low_ndsi,"GTiff")
#gdalout(inrast,os.path.join(path,"medium_ndsi_%s.tif" %date),medium_ndsi,"GTiff")
gdalout(inrast,os.path.join(path,"high_ndsi_toar_%s.tif" %date),high_ndsi,"GTiff")
