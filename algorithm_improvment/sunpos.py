import numpy as np
import math

from scipy import ndimage

data = np.array([[1, 1 , 1 , 1 , 1 , 1 , 0 , 1],
                 [1, 1 , 1 , 1 , 1 , 1 , 1 , 1],
                 [1, 1 , 0 , 0 , 0 , 0 , 1 , 0],
                 [0, 1 , 0 , 0 , 0 , 0 , 1 , 1],
                 [1, 1 , 0 , 0 , 0 , 0 , 0 , 1],
                 [1, 1 , 1 , 1 , 1 , 1 , 1 , 1],
                 [1, 1 , 1 , 1 , 1 , 1 , 1 , 1]])

kernel = [[1], [1], [1], [0], [0]]
print ndimage.binary_erosion(data, kernel, border_value=1).astype(np.uint8)
