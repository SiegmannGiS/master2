## Input File
# Tested under Windows 7 with Anaconda Python library

# Path for Images:
ImagesPath = "C:\Master\images/astental/"

# Correspondence File
CorrespondenceFile = "C:\Master\settings/astental/correspondence.txt"

# Specific DGM File for Correspondence File
DGM = "C:\Master\settings/astental/dgm_astental.asc"

# Specific Image for Correspondence File, Good Weather Image (blue sky)
CorrespondenceImage = "C:\Master\images/astental/ref-2016_08-24-1500.jpg"

# File Folder for saving Snow Classified Raster, Folder needs to exist
SnowClassFolder = "C:\Master\snowdetection/astental/"

# File Folder for saving Shadow Raster, Folder needs to exist
ShadowRastFolder = "C:\Master\shadows/astental/"

# Correspondence, Folder needs to exist
CorrondenceFolder = "C:\Master\correspondence/astental/"

# Temp Folder
Temp = "C:\Master/temp/"

# Use Mask?
mask = False
MaskFile = "C:\Master\settings/astental/mask.jpg"

# Method for Classifiying Snow
    # Options:
    #   1 = Corripio, Histogram Minima > blue value 127
    #   2 = Corripio, Shadow Snow version 2 Practise
SnowMethod = 2
tbl = 63
snowpixel = 127
# RGB Threshold if SnowMethod = 2
RGB =[127,127,127]

# SAGA Shadow Detection for improved Snow Detection
    # SAGA GIS has to be installed and Envoirenment Variables for SAGA has to be set
    # Option: True/False
    # Method 1 = Saga Snow detection
    # Method 2 = Grass Gis preprocessed

ShadowDetection = True
Method = 2
Latitude = 47.074531
longitude = 12.846210
timezone = 0
summertime = False

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
pathplot3 = "C:\Master\Image_with_snowdetection/astental"

# All important information
plot4 = True
saveplot4 = True
pathplot4 = "C:\Master\Image_with_snowdetection/astental"
