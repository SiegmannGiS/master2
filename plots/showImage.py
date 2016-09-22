from skimage.io import imread
import matplotlib.pyplot as plt

image = imread("D:\Master\Practise\PRACTISE_Matlabv2\Config_Astental\slr_ufs/test.jpg")


plt.imshow(image, interpolation="none")

y = int(image.shape[0]/2)
x = int(image.shape[1]/2)
plt.plot(x,y, "or")
plt.plot([image.shape[1]/2,image.shape[1]/2],[0,image.shape[0]])
plt.show()