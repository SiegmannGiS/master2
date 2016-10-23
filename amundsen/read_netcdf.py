from netCDF4 import Dataset, date2num, num2date
import numpy as np
import os
from datetime import datetime
from tools import ascii

path = "C:\Master/amundsen/astental/"
path2 = "C:\Master/amundsen/astental/results/2014"

ncdf = Dataset(os.path.join(path2,"OutputFields_2014.nc"), mode='r')


headinfo, dgm = ascii.read_ascii("C:\Master\settings/astental/dgm_astental.asc")

ncdf_swe = ncdf.variables['swe']
ncdf_time = ncdf.variables['time']
time_values = num2date(ncdf_time[:], ncdf_time.units)

dict = {date: ncdf_swe[date_num, :, :].squeeze().data for date_num, date in enumerate(time_values)}

print("done")
