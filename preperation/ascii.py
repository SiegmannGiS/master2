import numpy as np
import os


def read_ascii(path,file):
    """
    :param path: Input path
    :param file: Input file
    :return: headinfo as tuple, numpy data array
    """
    with open(os.path.join(path,file)) as fobj:
        header = fobj.readlines()[:6]
        ncols, nrows, xllcorner, yllcorner, cellsize, nodata = [float(x.split()[1]) for x in header]
        headinfo= (ncols, nrows, xllcorner, yllcorner, cellsize, nodata)
    dtm = np.loadtxt(os.path.join(path,file), skiprows=6)
    dtm[dtm == nodata] = np.nan
    return headinfo,dtm


def write_ascii(path,filename,headinfo,np_data,format="%f"):
    """
    :param path: Output path
    :param filename: Output filename
    :param headinfo: Tuple (ncols, nrows, xllcorner, yllcorner, cellsize, nodata)
    :param np_data: Numpy array
    :param format: default = "%f"
    :return: Esri ArcInfo grid
    """
    write_head = "ncols\t%s\nnrows\t%s\nxllcorner\t%s\nyllcorner\t%s\ncellsize\t%s\nnodata_value\t%s" %headinfo
    np.savetxt(os.path.join(path,filename),np_data,fmt=format,header=write_head,comments="")

