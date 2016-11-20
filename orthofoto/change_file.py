import os
import numpy as np
from tools import ascii

headinfo = ascii.read_ascii("dem_10m_noise.txt")[0]
correspondence = np.loadtxt("test.txt", delimiter=",")

new = np.transpose(correspondence[:4,:].astype(int))
# new = zip(new)
print(headinfo)

print(new)

with open(os.path.join("","ortho.txt"), "w") as fobj:
    for i, element in enumerate(new[::50]):

        y = (element[0] * headinfo[-2]) + headinfo[3]
        x = (element[1] * headinfo[-2]) + headinfo[2]
        fobj.write("%s -%s %s %s\n" %(element[2], element[3], x, y))