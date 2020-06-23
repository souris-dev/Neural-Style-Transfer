# The main file to train the network

import tensorflow as tf
import numpy as np
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
from tensorflow.keras.applications import vgg16

from models import ImageTransformNetwork
from losses import TheLoser
from traindata_prep import *
import argparse
import os
import requests

####################
# Argument Parsing
####################

parser = argparse.ArgumentParser()
parser.add_argument('model_checkpoint_prefix', type=str, default='./checkpoints/model',
                    help='Prefix for saving the model checkpoints')
parser.add_argument('--iter', type=int, default=40000, required=False,
                    help='Number of iterations to run.')
parser.add_argument('--content_weight', type=float, default=0.025, required=False,
                    help='Content weight.')
parser.add_argument('--style_weight', type=float, default=1.0, required=False,
                    help='Style weight.')
parser.add_argument('--tv_weight', type=float, default=1.0, required=False,
                    help='Total Variation weight.')
parser.add_argument('--pixel_weight', type=float, default=0.0, required=False,
                    help='Pixel loss weight')

args_in = parser.parse_args()
MODEL_SAVE_PREFIX = args_in.model_checkpoint_prefix

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
    # vgg16 performs mean subtraction on the pixels
    # (as a preprocessing step)
    img_arr = vgg16.preprocess_input(img_arr)
    return tf.convert_to_tensor(img_arr)


def deprocess_img(img_tensor: tf.Tensor) -> np.ndarray:
    img_arr = img_tensor.numpy()
    img_arr = img_arr.reshape((IMG_H, IMG_W, 3))
    
    # These are the mean pixel values for the ImageNet dataset
    # The VGG16 preprocessing subtracted these values, so we now add them
    img_arr[:,:, 0] += 103.939
    img_arr[:,:, 1] += 116.770
    img_arr[:,:, 2] += 123.68

    # Convert BGR to RGB
    img_arr = img_arr[:,:,::-1]  # That's just reversing the 3rd dimension
    img_arr = np.clip(img_arr, 0, 255).astype('uint8')
    return img_arr

#############################################################################

STYLE_WT = args_in.style_weight
FEATURE_RECONS_WT = args_in.content_weight
TV_WT = args_in.tv_weight
PIXEL_WT = args_in.pixel_weight

image_transform_net = ImageTransformNetwork()
image_transform_model = image_transform_net.model

fname = download_mscoco()
extract_mscoco(fname)

# We'll be training on the MS-COCO dataset

#base_img = preprocess_img(BASE_IMG_PATH)
#style_img = preprocess_img(STYLE_IMG_PATH)

#loss_measures = TheLoser(imshape=(IMG_H, IMG_W), STYLE_WT, FEATURE_RECONS_WT, TV_WT, PIXEL_WT)
#image_transform_net.compile_model(loss_measures)
#image_transform_net.train_model(base_img, style_img, loss_measures, MODEL_SAVE_PREFIX)

