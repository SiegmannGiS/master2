import scipy.spatial.distance as comp
import numpy as np


all_around = np.array([3,4,3,8,5,2,4,9,5,1,5,7])

match = np.array([3, 7, 4])
result = []
for i,num in enumerate(all_around):
    if i <= len(all_around)-len(match):
        compare = np.array(all_around[i:i+len(match)])
        #result.append(comp.correlation(compare,match))
        result.append(((compare - match) ** 2).mean())

print(result)
print(min(result))