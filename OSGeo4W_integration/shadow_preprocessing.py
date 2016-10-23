import os, sys
from datetime import datetime, timedelta
from pytz import timezone
import time

path = "C:\Master\images/astental"
path1 = "C:\Master\shadows/astental"

for i, name in enumerate(os.listdir(path)):
    if i > 0:
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
