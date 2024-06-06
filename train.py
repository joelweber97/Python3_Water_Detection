import tensorflow as tf
print(tf.__version__)
print(tf.config.list_physical_devices('GPU'))
import os
print(os.getcwd())
import rasterio as rio
import numpy as np
import matplotlib.pyplot as plt

naip_img = rio.open('naip22-nc-cir-60cm_2896204_20220528.tif')
print(naip_img.shape)
print(naip_img.height)
print(naip_img.width)
#np read (height, width, depth)
naip1 = naip_img.read(1)
naip2 = naip_img.read(2)
naip3 = naip_img.read(3)
naip4 = naip_img.read(4)

naip = np.dstack((naip1,naip2,naip3,naip4))


water_img = rio.open('naip22-nc-cir-60cm_2896204_20220528_water.tif')
print(water_img.shape)
print(water_img.height)
print(water_img.width)
#np reads (height, width, depth)
water = water_img.read(1)
water = np.expand_dims(water, axis = -1)


