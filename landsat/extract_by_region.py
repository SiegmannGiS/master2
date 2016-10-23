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
path2 = "C:\Master\landsat\layers"
scene_dic = {"astental":[191027, 192027], "patscherkofel":[192027, 193027], "vernagtferner":[193027]}

for region in scene_dic.keys():
    for scene in scene_dic[region]:
        for imagename in os.listdir(path2):
            if imagename[3:9] == str(scene):

                year = int(imagename[9:13])
                jday = int(imagename[13:16])
                date = datetime.datetime(year, 1, 1) + datetime.timedelta(jday)  # This assumes that the year is 2007
                type = imagename.split("_")[1].split(".")[0]
                newFilename = 'LC8_%s_%s.asc' %(date.strftime('%Y-%m-%d'),type)

                create_ascii(os.path.join(path2,imagename), "C:\Master\settings/%s/dgm_%s.asc" %(region,region),
                             os.path.join(path,region,"landsat",newFilename))




