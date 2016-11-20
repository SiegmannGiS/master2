import os
import shutil
import datetime

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

path = "C:\Master\images\patscherkofel"
for element in os.listdir(path):
    # old = element
    # element = element.split("_")
    # date = datetime.datetime(int(element[0]), int(element[1]), int(element[2]), int(element[3]),
    #                          int(element[4]))
    # date = roundTime(date, roundTo=60 * 60)
    # print(date)
    # element = datetime.datetime.strftime(date, "%Y-%m-%d_%H-%M")

    shutil.move(os.path.join(path,element), os.path.join(path,element[:-4]))