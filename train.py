import tensorflow as tf
print(tf.__version__)
print(tf.config.list_physical_devices('GPU'))
import os
print(os.getcwd())
import rasterio as rio
import numpy as np
import matplotlib.pyplot as plt
from patchify import patchify

import os
os.environ["SM_FRAMEWORK"] = "tf.keras"

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


#what size images should we use in the nn?
#possibly 256 pixel patches?

patch_size = 256
naip.shape[0]/256
naip.shape[1]/256
#50 patches x 44 patches
256 * 50
256 * 44

naip = naip[:12800,:11264,:]
water = water[:12800, :11264, :]



naip_patches = patchify(naip, (patch_size, patch_size, 4), step=patch_size) #Step=256 for 256 patches means no overlap
water_patches = patchify(water, (patch_size, patch_size, 1), step= patch_size)

naip_patches = naip_patches.reshape(2200, 256, 256, 4)
water_patches = water_patches.reshape(2200, 256, 256,1)

naip_patches = naip_patches/255.
water_patches = water_patches.astype('int32')



from tensorflow.keras.utils import to_categorical
labels_cat = to_categorical(water_patches, num_classes=2)

#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(naip_patches, labels_cat, test_size = 0.20, random_state = 42)


naip_patches.shape
labels_cat.shape


X_train = naip_patches[:178,:,:,:]
y_train = labels_cat[:178,:,:,:]

X_test = naip_patches[178:210,:,:,:]
y_test = labels_cat[178:210,:,:,:]




import segmentation_models as sm

weights = [0.5, 0.5]
dice_loss = sm.losses.DiceLoss(class_weights=weights) 
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)  #


IMG_HEIGHT = X_train.shape[1]
IMG_WIDTH  = X_train.shape[2]
IMG_CHANNELS = X_train.shape[3]
n_classes = 2

from std_unet import multi_unet_model, jacard_coef  

metrics=['accuracy', jacard_coef]

def get_model():
    return multi_unet_model(n_classes=n_classes, IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH, IMG_CHANNELS=IMG_CHANNELS)




model = get_model()
model.compile(optimizer='adam', loss=total_loss, metrics=metrics)
#model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=metrics)
model.summary()






history1 = model.fit(X_train, y_train, 
                    batch_size = 8, 
                    epochs=15, 
                    validation_data=(X_test, y_test), 
                    shuffle=True)


model.save('test.h5')
naip_patches = naip_patches[1:500,:,:,:]
water_patches = water_patches[1:500,:,:,:]
from random import randint

val = randint(0, len(naip_patches))
pred = model.predict(np.expand_dims(naip_patches[val], axis = 0))
predicted_img=np.argmax(pred, axis=3)[0,:,:]
print(np.unique(predicted_img, return_counts = True))
print(val)

plt.imsave(f'test_img_pred{val}.png', predicted_img)
plt.imsave(f'test_img{val}.png', naip_patches[val])