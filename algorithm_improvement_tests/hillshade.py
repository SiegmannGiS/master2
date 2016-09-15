import numpy as np
from preperation import ascii
import matplotlib.pyplot as plt
from astral import Astral
import datetime
import ephem


def assignBCs(elevGrid):
    # Pads the boundaries of a grid
    # Boundary condition pads the boundaries with equivalent values
    # to the data margins, e.g. x[-1,1] = x[1,1]
    # This creates a grid 2 rows and 2 columns larger than the input

    ny, nx = elevGrid.shape  # Size of array
    Zbc = np.zeros((ny + 2, nx + 2))  # Create boundary condition array
    Zbc[1:-1,1:-1] = elevGrid  # Insert old grid in center

    #Assign boundary conditions - sides
    Zbc[0, 1:-1] = elevGrid[0, :]
    Zbc[-1, 1:-1] = elevGrid[-1, :]
    Zbc[1:-1, 0] = elevGrid[:, 0]
    Zbc[1:-1, -1] = elevGrid[:,-1]

    #Assign boundary conditions - corners
    Zbc[0, 0] = elevGrid[0, 0]
    Zbc[0, -1] = elevGrid[0, -1]
    Zbc[-1, 0] = elevGrid[-1, 0]
    Zbc[-1, -1] = elevGrid[-1, 0]

    return Zbc


def calcFiniteSlopes(elevGrid, dx):
    # sx,sy = calcFiniteDiffs(elevGrid,dx)
    # calculates finite differences in X and Y direction using the
    # 2nd order/centered difference method.
    # Applies a boundary condition such that the size and location
    # of the grids in is the same as that out.

    # Assign boundary conditions
    Zbc = assignBCs(elevGrid)

    #Compute finite differences
    Sx = (Zbc[1:-1, 2:] - Zbc[1:-1, :-2])/(2*dx)
    Sy = (Zbc[2:,1:-1] - Zbc[:-2, 1:-1])/(2*dx)

    return Sx, Sy


def calcHillshade(elevGrid,dx,az,elev):
    #Hillshade = calcHillshade(elevGrid,az,elev)
    #Esri calculation for generating a hillshade, elevGrid is expected to be a numpy array

    # Convert angular measurements to radians
    azRad, elevRad = (360. - az + 90.)*np.pi/180., (90.-elev)*np.pi/180.
    Sx, Sy = calcFiniteSlopes(elevGrid, dx)  # Calculate slope in X and Y directions

    AspectRad = np.arctan2(Sy, Sx) # Angle of aspect
    SmagRad = np.arctan(np.sqrt(Sx**2. + Sy**2.))  # magnitude of slope in radians

    return (((np.cos(elevRad) * np.cos(SmagRad)) + (np.sin(elevRad)* np.sin(SmagRad) * np.cos(azRad - AspectRad))))*255

a = Astral()
Latitude = 47.074531
longitude = 12.846210

b = datetime.datetime(2015,10,3,7,30)

c= a.solar_azimuth(b,Latitude,longitude)
d= a.solar_elevation(b,Latitude,longitude)
print d, a.solar_zenith(b, Latitude, longitude)

print c

headinfo,dtm = ascii.read_ascii("C:\Master\settings/vernagtferner14-16/dgm_vernagtferner.txt")

test = calcHillshade(dtm,headinfo[-2],c,d)
print np.min(test), np.max(test)


plt.imshow(test, cmap="Greys")
plt.show()
ascii.write_ascii("test.asc",headinfo,test,format="%i")

