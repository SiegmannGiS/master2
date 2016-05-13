## Input File
# Tested under Windows 7 with Anaconda Python library

# Path for Images:
ImagesPath = "C:\Master\images/vernagtferner14-16/"

# Correspondence File
CorrespondenceFile = "C:\Master\settings/vernagtferner14-16/correspondence.txt"

# Specific DGM File for Correspondence File
DGM = "C:\Master\settings/vernagtferner14-16/dgm_vernagtferner.txt"

# Specific Image for Correspondence File, Good Weather Image (blue sky)
CorrespondenceImage = "C:\Master\images/vernagtferner14-16/2014-10-29_11-30_0.jpg"

# File Folder for saving Snow Classified Raster, Folder needs to exist
SnowClassFolder = "C:\Master\snowdetection/vernagtferner14-16/"

# File Folder for saving Shadow Raster, Folder needs to exist
ShadowRastFolder = "C:\Master\shadows/vernagtferner14-16/"

# Correspondence, Folder needs to exist
CorrondenceFolder = "C:\Master\correspondence/vernagtferner14-16/"

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
Latitude = 46.8720742
longitude = 10.822774
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
pathplot3 = "C:\Master\Image_with_snowdetection\Vernagtferner14-16"

