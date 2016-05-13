## Input File
# Tested under Windows 7 with Anaconda Python library

# Path for Images:
ImagesPath = "C:\Master\images/wallackhaus-nord/"

# Correspondence File
CorrespondenceFile = "C:\Master\settings/wallackhaus-nord/correspondence.txt"

# Specific DGM File for Correspondence File
DGM = "C:\Master\settings/wallackhaus-nord/dgm_wallackhaus.txt"

# Specific Image for Correspondence File, Good Weather Image (blue sky)
CorrespondenceImage = "C:\Master\images/wallackhaus-nord/2015-11-03_08-30_0.jpg"

# File Folder for saving Snow Classified Raster, Folder needs to exist
SnowClassFolder = "C:\Master\snowdetection/wallackhaus-nord/"

# File Folder for saving Shadow Raster, Folder needs to exist
ShadowRastFolder = "C:\Master\shadows/wallackhaus-nord/"

# Correspondence, Folder needs to exist
CorrondenceFolder = "C:\Master\correspondence/wallackhaus-nord/"

# Temp for Saga Files
Temp = "C:\Master/temp/"

# Method for Classifiying Snow
    # Options:
    #   1 = Corripio, Histogram Minima > blue value 127
    #   2 = Corripio, Shadow Snow version 2 Practise
SnowMethod = 2
tbl = 63
# RGB Threshold if SnowMethod = 2
RGB =[127,127,127]

# SAGA Shadow Detection for improved Snow Detection
    # SAGA GIS has to be installed and Envoirenment Variables for SAGA has to be set
    # Option: True/False
ShadowDetection = True
Latitude = 47.074531
longitude = 12.846210


# ColorCorrection of Shadow Areas
    # ShadowDetection has to be true
    # Option: True/False
ColorCorrection = False
     # 1 = additive: balancing the pixel intensities in the
     #     shadow and in the light by addition of the difference
     #     of pixel averages to the shadow pixels.
     # 2 = basiclightmodel: Simple crisp shadow removal derived
     #     from the light model containing a global, and a
     #     directed light.
ColorOption = 1


########################################################################################################################
##
## Visualization
##
########################################################################################################################

# Shadow as Image
    # Just for Presentation and Visualization
    # Option: True/False
ShadowAsImage = True

# blue color histogram/ classification plot
plot1 = False

# Height histogram plot
plot2 = False

# Classification points in Image
plot3 = False
saveplot3 = False
pathplot3 = "C:\Master\Image_with_snowdetection\wallackhaus-nord"


