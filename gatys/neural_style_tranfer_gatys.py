# Neural Style transfer
# (Optimization-based), using method by Gatys et al.

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, save_img
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
from tensorflow.keras import backend as K
from tensorflow.keras.applications import vgg19

import time
import argparse

####################
# Argument parsing
####################

parser = argparse.ArgumentParser(description='Neural Style Transfer (Gatys et al. method)')
parser.add_argument('base_image_path', metavar='base', type=str,
                    help='Path to the image to transform.')
parser.add_argument('style_reference_image_path', metavar='ref', type=str,
                    help='Path to the style reference image.')
parser.add_argument('result_prefix', metavar='res_prefix', type=str,
                    help='Prefix for the saved results.')
parser.add_argument('--iter', type=int, default=10, required=False,
                    help='Number of iterations to run.')
parser.add_argument('--content_weight', type=float, default=0.025, required=False,
                    help='Content weight.')
parser.add_argument('--style_weight', type=float, default=1.0, required=False,
                    help='Style weight.')
parser.add_argument('--tv_weight', type=float, default=1.0, required=False,
                    help='Total Variation weight.')
parser.add_argument('--optimizer', type=str, default='SGD',
                    help='Optimizer to use, can be SGD or L-BFGS')

args = parser.parse_args()

BASE_IMG_PATH = args.base_image_path
STYLE_IMG_PATH = args.style_reference_image_path
OUTPUT_PREFIX = args.result_prefix

# Weights for the loss components
tv_wt = args.tv_weight # 1e-6
style_wt = args.style_weight # 1e-6
content_weight = args.content_weight # 2.5e-8

# Picture dimensions we'll be dealing with
width, height = load_img(BASE_IMG_PATH).size
# Output image dimensions:
img_h = 400
img_w = int(width * img_h / height)


######################
# UTILITIES
# (utility functions)
######################

# Preprocessing and de-processing
# To convert tensor to image and vice-versa

def preprocess_img(img_path: str) -> tf.Tensor:
    img = load_img(img_path, target_size=(img_h, img_w))
    img_arr = img_to_array(img)
    img_arr = np.expand_dims(img_arr, axis=0)  # make the image one batch
    print('After expand_dims')
    # vgg19 performs mean subtraction on the pixels for normalization
    # (as a preprocessing step)
    img_arr = vgg19.preprocess_input(img_arr)
    return tf.convert_to_tensor(img_arr)

def deprocess_img(img_tensor: tf.Tensor) -> np.ndarray:
    img_arr = img_tensor.numpy()
    img_arr = img_arr.reshape((img_h, img_w, 3))
    
    # These are the mean pixel values for the ImageNet dataset
    # The VGG19 preprocessing subtracted these values, so we now add them
    img_arr[:,:, 0] += 103.939
    img_arr[:,:, 1] += 116.770
    img_arr[:,:, 2] += 123.68

    # Convert BGR to RGB
    img_arr = img_arr[:,:,::-1]  # That's just reversing the 3rd dimension
    img_arr = np.clip(img_arr, 0, 255).astype('uint8')
    return img_arr


####################
# LOSS FUNCTIONS
####################

# Let's define the loss functions now
# The stars of this show

# But first...  the gram matrix:

def gram_matrix(x: tf.Tensor) -> tf.Tensor:
    # make it channel-first
    x = tf.transpose(x, (2, 0, 1))

    # flatten it
    # Note: -1 implies keep the total number of elements same
    # So the above line serves to flatten x
    # for example, a tensor of shape (1, 3, 4)
    # would be made into (1, 12)
    feature_vector = tf.reshape(x, (x.shape[0], -1))

    # Calculate the gram matrix: G = V'V
    gram = tf.matmul(feature_vector, tf.transpose(feature_vector))
    return gram


# Now to calculate the style loss

def style_loss(y_pred: tf.Tensor, y_true: tf.Tensor) -> float:
    G_y_true = gram_matrix(y_true)
    G_y_pred = gram_matrix(y)
    channels = 3
    size = img_h * img_w

    # The formula in action
    return tf.reduce_sum(tf.square(G_y_true - G_y_pred) / (4 * (channels ** 2) * (size ** 2)))
    

# Content loss is simple MSE
# Lol we're not even taking mean here though

def content_loss(base: tf.Tensor, combo: tf.Tensor) -> float:
    return tf.reduce_sum(tf.square(combo - base))


# Total variation loss:
# Obtained by shifting the image pixels by 1 pixel down
# And by 1 pixel right

def tv_loss(x):
    # The tensor should be having rank of 4 here
    # 1 pixel shift down loss:
    a = tf.square(
        x[:, :img_h-1, :img_w-1, :] - x[:, 1:, :img_w-1, :]
    )
    # 1 pixel shift right loss:
    b = tf.square(
        x[:, :img_h-1, :img_w-1, :] - x[:, :img_h-1, 1:, :]
    )
    # combine a and b
    return tf.reduce_sum(tf.pow(a + b, 1.25))
    

############################
# FEATURE EXTRACTION
# (using pre-trained VGG19)
############################

# Let's get a VGG19 model from keras
model = vgg19.VGG19(include_top=False, weights='imagenet')

# Make a dictionary of layer names and the layer outputs
outputs_dict = {layer.name: layer.output for layer in model.layers}

# Now make a model that gives us output of each layer
feature_extractor = keras.Model(input=model.inputs, outputs=outputs_dict)

# The layers we'll be using to compute style loss (as suggested):
# The total style loss would be:
#  style loss weight * (sum of style loss for each layer) / no. of style loss layers
style_loss_layer_names = [
    "block1_conv1",
    "block2_conv1",
    "block3_conv1",
    "block4_conv1",
    "block5_conv1"
]

# Layer for content loss (as suggested):

content_loss_layer_name = "block5_conv2"

def compute_loss(combo, base, styleref):
    input_tensor = tf.concat([base, styleref, combo], axis=0)
    # get the outputs of the conv layers of VGG19
    features = feature_extractor(input_tensor)

    # init loss
    loss = tf.zeros(shape=())

    # Add content loss
    layer_content_features = features[content_loss_layer_name]
    base_content_features = layer_content_features[0,:,:,:]
    combo_content_features = layer_content_features[2,:,:,:]

    loss += content_weight * content_loss(base_content_features, combo_content_features)

    # Now add style loss

    for layer_name in style_loss_layer_names:
        layer_style_features = features[layer_name]
        base_style_eatures = layer_content_features[1,:,:,:]
        combo_style_features = layer_content_features[2,:,:,:]
        n_sty_layers = len(style_loss_layer_names)
        loss += style_weight * (style_loss(base_style_features, combo_style_features) / n_sty_layers)

    # Finally add the total variation loss
    loss += tv_wt * tv_loss(combo)

    return loss

# let's have a tf.function to compute losses and grads
# using tf.function to wrap it can make it faster

@tf.function
def compute_loss_and_grads(combo, base, styleref):
    with tf.GradientTape() as tape:
        loss = compute_loss(combo, base, styleref)
    grads = tape.gradient(loss, combo)
    return loss, grads


######################
# TRAINING
######################

ITERS = args.iter
OPTIMIZER = args.optimizer  # can be L-BFGS or SGD (stochastic gradient descent)

base_i = preprocess_img(BASE_IMG_PATH)
style_i = preprocess_img(STYLE_IMG_PATH)
combo_i = tf.Variable(initial_value=preprocess_img(BASE_IMG_PATH))

if OPTIMIZER == 'SGD':
    optimizer = keras.optimizers.SGD(
        keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=100, decay_steps=100, decay_rate=0.96
            )
        )

    for i in range(1, ITERS + 1):
        print('Iteration %d started, ' % i, end='')
        start_time = time.time()
        loss, grads = compute_loss_and_grads(combo_i, base_i, style_i)
        optimizer.apply_gradients([grads, combo_i])
        end_time = time.time()
        print('loss: %.2f' % loss)

        if i % 50 == 0:
            img = deprocess_img(combo_i)
            fname = OUTPUT_PREFIX + '_' + str(i) + '.jpg'
            save_img(fname, img)
        
        print('  Iteration %d took %ds' % (i, end_time - start_time))
        

# This may not work properly right now:
elif OPTIMIZER == 'L-BFGS':
    from scipy.optimize import fmin_l_bfgs_b

    # An object of this class allows the fmin_l_bfgs_b function to 
    # retrieve loss and grads by calling the
    # compute_loss_and_grads function only once rather than
    # two times separately to get the loss and grads
    
    class Evaluator(object):
        def __init__(self):
        self.loss_value = None
        self.grads_values = None

        def get_loss_and_grads_ndarr(x):
            return compute_loss_and_grads(tf.convert_to_tensor(x))

        def get_loss(self, x):
            assert self.loss_value is None
            loss_value, grad_values = get_loss_and_grads_ndarr(x)
            self.loss_value = loss_value
            self.grad_values = grad_values
            return self.loss_value

        def get_grads(self, x):
            assert self.loss_value is not None
            grad_values = np.copy(self.grad_values)
            self.loss_value = None
            self.grad_values = None
            return grad_values

    evaluator = Evaluator()

    for i in range(1, ITERS + 1):
        print('Iteration %d started, ' % i, end='')
        start_time = time.time()

        combo_i, _mins, _inf = fmin_l_bfgs_b(evaluator.loss, combo_i.numpy().flatten(), fprime=evaluator.grads)
        end_time = time.time()
        print('loss: %.2f' % loss)

        if i % 50 == 0:
            img = deprocess_img(tf.convert_to_tensor(combo_i))
            fname = OUTPUT_PREFIX + '_' + str(i) + '.jpg'
            save_img(fname, img)
        
        print('  Iteration %d took %ds' % (i, end_time - start_time))