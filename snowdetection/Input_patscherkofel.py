## Input File
# Tested under Windows 7 with Anaconda Python library

# Path for Images:
ImagesPath = "C:\Master\images/patscherkofel/"

# Correspondence File
CorrespondenceFile = "C:\Master\settings/patscherkofel/correspondence.txt"

# Specific DGM File for Correspondence File
DGM = "C:\Master\settings/patscherkofel/dgm_patscherkofel.asc"

# Specific Image for Correspondence File, Good Weather Image (blue sky)
CorrespondenceImage = "C:\Master\settings/patscherkofel/img2015_03_23_13_59_59.jpg"

# File Folder for saving Snow Classified Raster, Folder needs to exist
SnowClassFolder = "C:\Master\snowdetection/patscherkofel/"

# File Folder for saving Shadow Raster, Folder needs to exist
ShadowRastFolder = "C:\Master\shadows/patscherkofel/"

# Correspondence, Folder needs to exist
CorrondenceFolder = "C:\Master\correspondence/patscherkofel/"

# hillshade
Hillshade = "C:\Master\settings/patscherkofel\hillshade.asc"

# extent
extent = (250519, 266449, 363962, 375972)

# Temp Folder
Temp = "C:\Master/temp/"

# Use Mask?
mask = True
MaskFile = "C:\Master\settings/patscherkofel/mask.asc"

# Method for Classifiying Snow
    # Options:
    #   1 = Corripio, Histogram Minima > blue value 127
    #   2 = Corripio, Shadow Snow version 2 Practise
SnowMethod = 1
tbl = 63
snowpixel = 127
# RGB Threshold if SnowMethod = 2
RGB =[127,127,127]

# SAGA Shadow Detection for improved Snow Detection
    # SAGA GIS has to be installed and Envoirenment Variables for SAGA has to be set
    # Option: True/False
    # Method 1 = Saga Snow detection
    # Method 2 = Grass Gis preprocessed

ShadowDetection = False
Method = 2

# for Method 1
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
ShadowAsImage = False

# blue color histogram/ classification plot
plot1 = False

# Height histogram plot
plot2 = False

# Classification points in Image
plot3 = False
saveplot3 = False
pathplot3 = "C:\Master\Image_with_snowdetection/patscherkofel"

# All important information
plot4 = True
saveplot4 = True
pathplot4 = "C:\Master\Image_with_snowdetection/patscherkofel"