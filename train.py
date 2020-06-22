# The main file to train the network

import tensorflow as tf
import numpy as np
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
from tensorflow.keras.applications import vgg16

from models import ImageTransformNetwork
from losses import feature_loss, style_loss, tv_loss
import argparse

BASE_IMG_PATH = ''
STYLE_IMG_PATH = ''
OUTPUT_PREFIX = ''

MODEL_SAVE_PREFIX = ''

######################
# Image Utilities
######################

width, height = load_img(BASE_IMG_PATH).size
# Output image dimensions:
IMG_H = 400
IMG_W = int(width * img_h / height)

def preprocess_img(img_path: str) -> tf.Tensor:
    img = load_img(img_path, target_size=(IMG_H, IMG_W))
    img_arr = img_to_array(img)
    img_arr = np.expand_dims(img_arr, axis=0)  # make the image one batch
    print('After expand_dims')
    # vgg19 performs mean subtraction on the pixels for normalization
    # (as a preprocessing step)
    img_arr = vgg19.preprocess_input(img_arr)
    return tf.convert_to_tensor(img_arr)


def deprocess_img(img_tensor: tf.Tensor) -> np.ndarray:
    img_arr = img_tensor.numpy()
    img_arr = img_arr.reshape((IMG_H, IMG_W, 3))
    
    # These are the mean pixel values for the ImageNet dataset
    # The VGG19 preprocessing subtracted these values, so we now add them
    img_arr[:,:, 0] += 103.939
    img_arr[:,:, 1] += 116.770
    img_arr[:,:, 2] += 123.68

    # Convert BGR to RGB
    img_arr = img_arr[:,:,::-1]  # That's just reversing the 3rd dimension
    img_arr = np.clip(img_arr, 0, 255).astype('uint8')
    return img_arr


image_transform_net = ImageTransformNetwork()
image_transform_model = image_transform_net.model
