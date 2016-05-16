from netCDF4 import Dataset, date2num, num2date
import numpy as np
import os
import preperation.ascii as ascii
from datetime import datetime

path = "C:\Master/amundsen/vernagtferner14-16/"


fh = Dataset(os.path.join(path,"OutputFields_2015.nc"), mode='r')


print fh
headinfo= (1131, 985, 205792.50000001, 330852.5, 5, -9999)


for i in range(len(fh.variables["time"])):
    try:
        time = num2date(fh.variables["time"][i], fh.variables["time"].units)
        time = datetime.strftime(time, "%Y-%m-%d_%H-%M.txt")
        print(time)
        ascii.write_ascii(os.path.join(path,"data"),time,headinfo,fh.variables["swe"][i], format="%.2f")
    except:
        print("%s missing" %i)