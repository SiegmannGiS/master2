import numpy as np
import cv2

img = cv2.imread('C:\Master\images\patscherkofel/2014-12-23_09-00.jpg',0)
equ = cv2.equalizeHist(img)
res = np.hstack((img,equ)) #stacking images side-by-side
cv2.imwrite('res.png',res)