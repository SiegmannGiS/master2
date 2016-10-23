import grass.script as gscript
import grass.script.setup as gsetup
import os
from grass import script
from datetime import datetime

# Init grass
gisbase = r"C:\OSGEO4~1\apps\grass\grass-7.0.4"
gisdb = r"C:\grassdb"
mapset = "PERMANENT"
location = "vernagtferner"
gsetup.init(gisbase, gisdb, location, mapset)

# set region
script.run_command("g.region", rast="dem")
print("region set")
#
#Paths
path = "C:\Master\images/%s" %location
path1 = "C:\Master\shadows/%s" %location

# iterate over date of images
for i, name in enumerate(os.listdir(path)):
    #if i > 0:
    if name == "2014-10-25_10-00.jpg":
        print("\nProgress: %.2f" % (i / float(len(os.listdir(path))) * 100))
        date = datetime.strptime(name.split(".")[0], "%Y-%m-%d_%H-%M")
        civil_time = "0"
        yday = date.timetuple().tm_yday
        Minute = "%i" % (int(date.minute) / 60. * 100.)
        Moment = "%s.%s" % (date.hour, Minute)
        os.system("r.sun elevation=dem aspect=dem_aspect slope=dem_slope lat=dem_lat long=dem_lon incidout=shadow"
              " civil_time=%s day=%s time=%s --overwrite" % (civil_time, yday, Moment))
        os.system("r.out.ascii input=shadow output=%s.asc null_value=-9999.0 -h --overwrite" % (
        os.path.join(path1, name.split(".")[0])))


# name = "2016-07-10_05-32.jpg"
# date = datetime(2016,7,10,5,32)
# civil_time = "0"
# yday = date.timetuple().tm_yday
# Minute = "%i" % (int(date.minute) / 60. * 100.)
# Moment = "%s.%s" % (date.hour, Minute)
# os.system("r.sun elevation=dem aspect=dem_aspect slope=dem_slope lat=dem_lat long=dem_lon incidout=shadow"
#           " civil_time=%s day=%s time=%s --overwrite" % (civil_time, yday, Moment))
# os.system("r.out.ascii input=shadow output=%s.asc null_value=-9999.0 -h --overwrite" % (
#     os.path.join(path1, name.split(".")[0])))