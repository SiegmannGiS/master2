import grass.script as gscript
import grass.script.setup as gsetup
import os

###########
# launch session
gisbase = r"C:\OSGEO4~1\apps\grass\grass-7.0.4"
gisdb = r"C:\grassdb"
mapset = "PERMANENT"

for i,location in enumerate(sorted(os.listdir(gisdb))):
    print("\n\n"+location)
    gsetup.init(gisbase,gisdb, location, mapset)

    gscript.message('Current GRASS GIS 7 environment:')
    print(gscript.gisenv())

    gscript.message('Available raster maps:')
    for rast in gscript.list_strings(type='rast'):
        print(rast)

    gscript.message('Available vector maps:')
    for vect in gscript.list_strings(type='vect'):
        print(vect)

    print("done")