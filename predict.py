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

from tensorflow.keras.utils import to_categorical

import segmentation_models as sm
from std_unet import multi_unet_model, jacard_coef  
from random import randint




naip_img = rio.open('naip22-nc-cir-60cm_2896204_20220528.tif')
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


patch_size = 256
naip.shape[0]/256
naip.shape[1]/256
#50 patches x 44 patches
256 * 50
256 * 44

naip = naip[:12800,:11264,:]


naip_patches = patchify(naip, (patch_size, patch_size, 4), step=patch_size) #Step=256 for 256 patches means no overlap

naip_patches = naip_patches.reshape(2200, 256, 256, 4)

naip_patches = naip_patches/255.






import segmentation_models as sm
weights = [0.5, 0.5]
dice_loss = sm.losses.DiceLoss(class_weights=weights) 
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)  #
from std_unet import multi_unet_model, jacard_coef  
model = tf.keras.models.load_model('test.h5', custom_objects = {'dice_loss_plus_1focal_loss': total_loss, 'jacard_coef': jacard_coef})




preds = []
count = 0
for i in naip_patches:
    plt.imsave(f'original_images/image{count}.png', i)
    pred = np.argmax(model.predict(np.expand_dims(i, axis = 0)), axis = -1).reshape(256,256)
    plt.imsave(f'predicted_images/image{count}.png', pred)
    preds.append(pred)
    count +=1


'''
preds = np.array(preds)


preds = preds.reshape(50,44,256,256)


del(naip_patches)
del(naip)
del(model)


up = unpatchify(preds, (12800, 11264))
del(preds)

'''