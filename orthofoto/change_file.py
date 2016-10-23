import os
import numpy as np
from tools import ascii

filepath = "C:\Master\settings/astental\correspondence.txt"
path = "C:\Master\settings/astental"
headinfo = ascii.read_ascii("C:\Master\settings/astental/dgm_astental.asc")[0]
correspondence = np.loadtxt(filepath, delimiter=",")

new = np.transpose(correspondence[:4,:].astype(int))
# new = zip(new)
print(headinfo)

print(new)

with open(os.path.join(path,"ortho.txt"), "w") as fobj:
    for i, element in enumerate(new[::20]):

        y = (element[0] * headinfo[-2]) + headinfo[3]
        x = (element[1] * headinfo[-2]) + headinfo[2]
        fobj.write("%s -%s %s %s\n" %(element[2], element[3], x, y))