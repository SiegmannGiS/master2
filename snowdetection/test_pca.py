import numpy as np
from numpy import genfromtxt
from numpy import matlib

arrayview = genfromtxt("E:\Master\Practise\PRACTISE_Matlabv2\PRACTISE_V_2_0/test.txt", delimiter=",")


rgb=rgbimage[4:7,:].T

print rgb
print rgbimage.shape