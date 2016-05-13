import numpy as np

a = np.array([[1,2,3,4],[2,3,4,5],[1,0,1,0]])

print a

cond = np.where(a[2,:] == 1)[0]

print a[:,cond]