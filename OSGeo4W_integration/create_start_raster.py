import grass.script as gscript
import grass.script.setup as gsetup
import os
from grass import script
import argparse

parser = argparse.ArgumentParser(description="Gebiet", usage="python create_start_raster.py -site vernagtferner")
parser.add_argument("-site", type=str, help="study site")

args = parser.parse_args()
loca = args.site


gisbase = r"C:\OSGEO4~1\apps\grass\grass-7.0.4"
gisdb = r"C:\grassdb"
mapset = "PERMANENT"

for i,location in enumerate(sorted(os.listdir(gisdb))):
    if location == loca:
        print(location)
        # log in location
        gsetup.init(gisbase, gisdb, location, mapset)

        # import raster
        script.run_command("r.in.gdal", input="C:\Master\settings\\%s\dgm_%s.asc" %(loca,loca),
                           output="dem", overwrite=True, flags="oe")

        # set region
        script.run_command("g.region", rast="dem")

        # # calculate lat / lon
        script.run_command("r.latlong", input="dem",output="dem_lat", overwrite=True)
        script.run_command("r.latlong", input="dem", output="dem_lon", overwrite=True, flags="l")

        # slope and aspect
        script.run_command("r.slope.aspect", elevation="dem", aspect="dem_aspect", slope="dem_slope", overwrite=True)




