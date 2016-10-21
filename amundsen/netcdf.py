from netCDF4 import Dataset, date2num, num2date
import numpy as np
import os
from datetime import datetime


class ReadNetCDF(object):
    def __init__(self, filename):
        self.fobj = Dataset(filename, mode='r')
        self.variables = self.fobj.variables
        self.dimensions = self.fobj.dimensions

    def close(self):
        self.fobj.close()

    def get_time_dict_by_attribute(self, attr):
        if len(self.variables[attr].shape) == 3:
            ncdf_attr = self.variables[attr]
            ncdf_time = self.variables['time']
            time_values = num2date(ncdf_time[:], ncdf_time.units)
            timedict = {date: ncdf_attr[date_num, :, :].squeeze().data for date_num, date in enumerate(time_values)}

            return timedict
        else:
            print("Attribute does not have a time dimension")

    def get_array_from_datetime(self, datetime_obj, attr):
        if len(self.variables[attr].shape) == 3:
            date_as_num = date2num(datetime_obj,units=self.variables["time"].units)
            ncdf_attr = self.variables[attr]
            return ncdf_attr[self.variables['time'][:] == date_as_num, :, :].squeeze().data

        else:
            print("Attribute does not have a time dimension")


class WriteNetCDF(object):
    def __init__(self, filename):
        self.fobj = Dataset(filename, mode='r')
        if not os.path.exists(filename):
            print("Dimensions and Variables have to be added with .add_meta")