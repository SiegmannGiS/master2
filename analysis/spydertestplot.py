import numpy as np
import pylab as plt

fig = plt.figure()
ax = plt.axes(polar=True)
import random
r = random.sample(xrange(361), 10)
theta = np.pi/360 * np.array(list(range(0, 360)))

ax.plot(theta, r, "r-")
#ax.errorbar(theta, r, yerr=1, xerr=.1, capsize=1)

print r
print theta

plt.show()