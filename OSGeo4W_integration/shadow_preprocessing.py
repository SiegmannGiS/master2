import os
from datetime import datetime

path = "C:\Master\images\wallackhaus-nord"
path1="C:\Master\shadows\wallackhaus-nord"

length = float(len(os.listdir(path)))

for i,name in enumerate(os.listdir(path)):
    if i > 0:
        print("\nProgress: %.2f" %(i/length*100))
        ImageInfo = "_".join(name.split("_")[:2])
        date = datetime.strptime(ImageInfo, "%Y-%m-%d_%H-%M")
        yday = date.timetuple().tm_yday
        Minute = "%i" % (int(date.minute) / 60. * 100.)
        Moment = "%s.%s" % (date.hour, Minute)
        print("r.sun elevation=dem aspect=dem_aspect slope=dem_slope lat=dem_lat long=dem_lon incidout=shadow"
                  " civil_time=+1 day=%s time=%s --overwrite" %(yday,Moment))
        print("r.out.ascii input=shadow output=%s.asc null_value=-9999.0 -h --overwrite" %(os.path.join(path1, name.split(".")[0])))
