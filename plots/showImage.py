from skimage.io import imread
import matplotlib.pyplot as plt

image = imread("C:\Master\images\wallackhaus-nord/2015-10-05_11-30_0.jpg")


plt.imshow(image)

y = int(image.shape[0]/2)
x = int(image.shape[1]/2)
plt.plot(x,y, "or")
plt.plot([image.shape[1]/2,image.shape[1]/2],[0,image.shape[0]])
plt.show()