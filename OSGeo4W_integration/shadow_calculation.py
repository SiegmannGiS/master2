import grass.script as gscript
import grass.script.setup as gsetup
import os
from grass import script
from snowdetection import Input_vernagtferner as Input
from datetime import datetime
import numpy as np

# Init grass
gisbase = r"C:\OSGEO4~1\apps\grass\grass-7.0.4"
gisdb = r"C:\grassdb"
mapset = "PERMANENT"
location = "vernagtferner"
gsetup.init(gisbase, gisdb, location, mapset)

# set region
script.run_command("g.region", rast="dem")
print("region setted")

# iterate over date of images
for i,name in enumerate(os.listdir(Input.ImagesPath)):
    if i == 0:
        ImageInfo = "_".join(name.split("_")[:2])
        date = datetime.strptime(ImageInfo, "%Y-%m-%d_%H-%M")
        yday = date.timetuple().tm_yday
        Minute = "%i" % (int(date.minute) / 60. * 100.)
        Moment = "%s.%s" % (date.hour, Minute)
        print("start grass command")
        script.run_command("r.sun", elevation="dem", aspect="dem_aspect", slope="dem_slope", lat="dem_lat",
                           long="dem_lon", incidout="shadow", civil_time=+1, day=yday, time=Moment, overwrite=True)
        print("done")