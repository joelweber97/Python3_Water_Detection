import tensorflow as tf
print(tf.__version__)
print(tf.config.list_physical_devices('GPU'))
import os
print(os.getcwd())
import rasterio as rio
import numpy as np
import matplotlib.pyplot as plt
from patchify import patchify, unpatchify
import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
import math
from tensorflow.keras.utils import to_categorical

import segmentation_models as sm
from std_unet import multi_unet_model, jacard_coef  
from random import randint





naip_img = rio.open('naip22-nc-cir-60cm_2896204_20220528.tif')
#i think we need to window this dataset down to the actual output size.


profile = naip_img.profile
prj = naip_img.crs
gt = naip_img.transform
print(naip_img.shape)
print(naip_img.height)
print(naip_img.width)
#np read (height, width, depth)
naip1 = naip_img.read(1)
naip2 = naip_img.read(2)
naip3 = naip_img.read(3)
naip4 = naip_img.read(4)

naip = np.dstack((naip1,naip2,naip3,naip4))
del(naip1, naip2, naip3, naip4)


patch_size = 256

y = math.ceil(naip.shape[0]/patch_size)
x = math.ceil(naip.shape[1]/patch_size)

y = y * patch_size
x = x * patch_size

z = np.zeros(shape = (y, x, naip.shape[2]))
print(z.shape)

z[:naip.shape[0], :naip.shape[1], :] = naip
z.shape


#naip_patches = patchify(naip, (patch_size, patch_size, 4), step=patch_size) #Step=256 for 256 patches means no overlap
naip_patches = patchify(z, (patch_size, patch_size, 4), step=patch_size) #Step=256 for 256 patches means no overlap

naip_patches = naip_patches[:,:,0,:,:,:]

naip_patches = naip_patches/255.


import segmentation_models as sm
weights = [0.5, 0.5]
dice_loss = sm.losses.DiceLoss(class_weights=weights) 
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)  #
from std_unet import multi_unet_model, jacard_coef  
model = tf.keras.models.load_model('test.h5', custom_objects = {'dice_loss_plus_1focal_loss': total_loss, 'jacard_coef': jacard_coef})




patched_prediction = []
for i in range(naip_patches.shape[0]):
    for j in range(naip_patches.shape[1]):
        
        single_patch_img = naip_patches[i,j,:,:,:]
        #print(single_patch_img.shape)
        #Use minmaxscaler instead of just dividing by 255. 
        #single_patch_img = scaler.fit_transform(single_patch_img.reshape(-1, single_patch_img.shape[-1])).reshape(single_patch_img.shape)
        single_patch_img = np.expand_dims(single_patch_img, axis=0)
        pred = model.predict(single_patch_img)
        pred = np.argmax(pred, axis=3)
        pred = pred[0, :,:]         
        patched_prediction.append(pred)

patched_prediction = np.array(patched_prediction)
patched_prediction = np.reshape(patched_prediction, [naip_patches.shape[0], naip_patches.shape[1], 
                                            naip_patches.shape[2], naip_patches.shape[3]])




unpatched_prediction = unpatchify(patched_prediction, (z.shape[0], z.shape[1]))

unpatched_pred = unpatched_prediction[:naip.shape[0], :naip.shape[1]]

#unpatched_pred = np.expand_dims(unpatched_pred, axis = -1)

profile['count'] = 1

with rio.open('up_with_projection.tif', 'w', **profile) as dst:
    dst.write(unpatched_pred, 1)



#faily decent predictions to this point.