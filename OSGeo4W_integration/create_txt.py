import os
import datetime
location = "astental"
path = "C:\Master\images/%s" %location

def roundTime(dt=None, roundTo=60):
   """Round a datetime object to any time laps in seconds
   dt : datetime.datetime object, default now.
   roundTo : Closest number of seconds to round to, default 1 minute.
   Author: Thierry Husson 2012 - Use it as you want but don't blame me.
   """
   if dt == None : dt = datetime.datetime.now()
   seconds = (dt.replace(tzinfo=None) - dt.min).seconds
   rounding = (seconds+roundTo/2) // roundTo * roundTo
   return dt + datetime.timedelta(0,rounding-seconds,-dt.microsecond)


with open("%s.txt" % location, "w") as fobj_out:
    for element in os.listdir(path):
        element = element.split(".")[0]

        # for patscherkofel
        # element = element.split("_")
        # date = datetime.datetime(int(element[0][3:]), int(element[1]), int(element[2]), int(element[3]),
        #                          int(element[4]))
        # date = roundTime(date, roundTo=60 * 60)
        # element = datetime.datetime.strftime(date, "%Y-%m-%d_%H-%M")

        fobj_out.write("%s\n" %element)