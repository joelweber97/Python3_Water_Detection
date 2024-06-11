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

from tensorflow.keras.utils import to_categorical

import segmentation_models as sm
from std_unet import multi_unet_model, jacard_coef  
from random import randint


