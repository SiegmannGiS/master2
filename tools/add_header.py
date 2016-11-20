import ascii
import numpy as np
import os

path = "C:\Master\shadows/vernagtferner"

headinfo = ascii.read_ascii("C:\Master\settings/vernagtferner/dgm5_vernagtferner.asc")[0]

for element in os.listdir(path):
    array = np.loadtxt(os.path.join(path, element))
    ascii.write_ascii(os.path.join(path,element), headinfo, array)