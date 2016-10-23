from netCDF4 import Dataset
import numpy as np
from numpy.random import uniform

dataset = Dataset('test.nc', "w", format='NETCDF4_CLASSIC')

# Create dimensions
x = dataset.createDimension('x', 4)
y = dataset.createDimension('y', 3)
time = dataset.createDimension('time', None)

# Create variables
times = dataset.createVariable('time', np.float64, ('time'))
x_var = dataset.createVariable("x", np.float32, ("x"))
y_var = dataset.createVariable("y", np.float32, ("y"))
snow_classification = dataset.createVariable("snow", np.int32, ("time","y", "x"))

# Variable attributes
times.units = "hours since 0001-01-01 00:00:00"
x_var.units = "m"
x_var.long_name = "x coordinate of projection"
y_var.units = "m"
y_var.long_name = "y coordinate of projection"
snow_classification.units = "bool"
snow_classification.nan = -9999

print(dataset)