"""
===============================
Piecewise Affine Transformation
===============================

This example shows how to use the Piecewise Affine Transformation.

"""
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import PiecewiseAffineTransform, warp
from skimage.io import imread


image = imread("C:\Master\images/vernagtferner14-16/2014-11-21_16-00_35.jpg")

with np.load("C:\Master\settings/vernagtferner14-16/correspondence.txtcor.npz") as data:
    cor = data["cor"]

# index of raster cell, which can be seen
cond = np.where(cor[7,:] == 1)[0]
arrayview = cor[:,cond]
arrayview = arrayview[:4,:]



src_rows, src_cols = np.meshgrid(arrayview[1,::1000],arrayview[0,::1000])


src = np.dstack([src_cols.flat, src_rows.flat])[0]

# add sinusoidal oscillation to row coordinates
dst_cols = arrayview[2,::1000]
dst_rows = arrayview[3,::1000]
dst = np.vstack([dst_cols, dst_rows]).T




tform = PiecewiseAffineTransform()
tform.estimate(src, dst)


out = warp(image, tform)

fig, ax = plt.subplots()
ax.imshow(out)
plt.show()
