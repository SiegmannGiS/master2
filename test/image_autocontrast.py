from PIL import ImageEnhance
from PIL import Image
import matplotlib.pyplot as plt

path = "C:\Master\images/vernagtferner14-16/2014-11-26_15-30_0.jpg"

image = Image.open(path)

newimage = ImageEnhance.Color(image)
newimage.enhance(2).show()
image.show()




