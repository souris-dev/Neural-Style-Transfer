# The main file to train the network

import tensorflow as tf
import numpy as np
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
from tensorflow.keras.applications import vgg16
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from models import ImageTransformNetwork
from losses import LossGenerator
from traindata_prep import *
import argparse
import os
import requests

####################
# Argument Parsing
####################

parser = argparse.ArgumentParser()
parser.add_argument('style_image', type=str, help='Style reference image')
parser.add_argument('model_checkpoint_prefix', type=str, default='./checkpoints/model',
                    help='Prefix for saving the model checkpoints')
parser.add_argument('--epochs', type=int, default=2, required=False,
                    help='Number of epochs to run.')
parser.add_argument('--content_weight', type=float, default=1.0, required=False,
                    help='Content weight.')
parser.add_argument('--style_weight', type=float, default=1.0, required=False,
                    help='Style weight.')
parser.add_argument('--tv_weight', type=float, default=1e-6, required=False,
                    help='Total Variation weight.')
parser.add_argument('--download_folder', type=str, required=True, help='Download folder for dataset')
parser.add_argument('--pixel_weight', type=float, default=0.0, required=False,
                    help='Pixel loss weight')

args_in = parser.parse_args()
STYLE_IMG_PATH = args_in.style_image
MODEL_SAVE_PREFIX = args_in.model_checkpoint_prefix

######################
# Image Utilities
######################

# width, height = load_img(BASE_IMG_PATH).size
# Output image dimensions:
IMG_H = 256
IMG_W = 256

def preprocess_style_img(img_path: str) -> np.ndarray:
    img = load_img(img_path, target_size=(IMG_H, IMG_W))
    img_arr = img_to_array(img)
    #img_arr = np.expand_dims(img_arr, axis=0)  # make the image one batch
    # vgg16 performs mean subtraction on the pixels
    # (as a preprocessing step)
    img_arr = vgg16.preprocess_input(img_arr)
    return img_arr


def deprocess_img(img_tensor: np.ndarray) -> np.ndarray:
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


# Let's download and prepare the data
fname = download_mscoco(download_folder=args_in.download_folder)
dataset_directory = extract_mscoco(fname)

# Prepare the data generator
data_generator = ImageDataGenerator(preprocessing_function=vgg16.preprocess_input)
BATCH_SIZE = 4
data_flow = data_generator.flow_from_directory(directory=train_directory,
    batch_size=BATCH_SIZE,
    class_mode='input',
    target_size=(256, 256)
)

# Compile and train the image transfor network
loss_gen = LossGenerator(
    imshape=(256, 256), style_wt=STYLE_WT, feature_recons_wt=FEATURE_RECONS_WT,
    tv_wt=TV_WT, pixel_wt=PIXEL_WT
)

image_transform_net = ImageTransformNetwork(style_img=preprocess_style_img(STYLE_IMG_PATH))
print('\Summary of the architecture of ImageTranformNetwork: ')
image_transform_net.model.summary()

image_transform_net.compile_model(loss_gen)
image_transform_net.train_model(data_flow, epochs=args_in.epochs, model_checkpoint_prefix=MODEL_SAVE_PREFIX)

print('Training finished!')
print('Saving model to: model.h5')
print('Saving model_weights to: model_weights.h5')

image_transform_net.save_model('model.h5')
image_transform_net.save_model_weights('model_weights.h5')

# Also save in the Protocol Buffers format
print('Saving model also as: model.pb')
image_transform_net.save_model_pb('model_frozen_graph.pb')