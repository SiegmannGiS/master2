import arcpy
from arcpy.sa import *
from arcpy import env
import os, datetime
# Change Interpreter to arcpy in PyCharm for running this script

def create_ascii(inrast,mask,outrast):
    # Execute ExtractByMask
    outExtractByMask = ExtractByMask(inrast, mask)
    arcpy.Resample_management(outExtractByMask, "resample", "10", "NEAREST")
    # Save the output
    arcpy.RasterToASCII_conversion("resample", outrast)


# settings
env.workspace = "C:\Users\Marcel\Documents\ArcGIS\Default.gdb"
env.overwriteOutput = True
arcpy.env.outputCoordinateSystem = arcpy.SpatialReference("MGI Austria Lambert")
arcpy.env.geographicTransformations = "MGI_To_ETRS_1989_5; ETRS_1989_To_WGS_1984"
arcpy.CheckOutExtension("Spatial")


# Main
path = "C:\Master\landsat"


for region in os.listdir(path):

    mask = "AOI_"+region.title()
    for element in os.listdir(os.path.join(path,region,"images")):
        if element[-4:] == ".png":
            print(element)
            year = int(element[9:13])
            jday = int(element[13:16])
            date = datetime.datetime(year, 1, 1) + datetime.timedelta(jday)  # This assumes that the year is 2007
            newFilename = 'lc8_%s.asc' % date.strftime('%Y-%m-%d')

            create_ascii(os.path.join(path,region,"images",element), "C:\Master\settings.gdb/%s" %(mask),
                         os.path.join(path,region,"ascii",newFilename))




