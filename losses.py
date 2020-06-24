# Loss Functions

import tensorflow as tf
import numpy as np

from tensorflow.keras.applications import vgg16
from tensorflow.keras.models import Model

from functools import partial

# This class has all losses we'll use, so it's LossGenerator
# Don't get me wrong, it's still a star of the show ;-)
class LossGenerator:
    """
    This class groups all the loss methods together
    So that one single Loser object can be created for a given image size
    """

    def __init__(self, imshape: tuple, style_wt: float, feature_recons_wt: float, tv_wt: float, pixel_wt=0.0):
        self.img_h = imshape[0]
        self.img_w = imshape[1]

        # weights for the respective losses
        self.style_wt = style_wt  # style loss wt.
        self.feature_recons_wt = feature_recons_wt  # feature reconstruction loss wt.
        self.tv_wt = tv_wt  # total variation loss wt.
        self.pixel_wt = pixel_wt  # pixel loss weight

        # Here are the models and layers we'll use
        # for our perceptual loss(es)
        self.loss_model_vgg16 = vgg16.VGG16(include_top=False, weights='imagenet')
        self.layers_vgg16 = {layer.name: layer.output for layer in self.loss_model_vgg16.layers}

        self.style_layers = [
            'block1_conv2',
            'block2_conv2',
            'block3_conv3',
            'block4_conv3'
        ]

        self.feature_recons_layer = 'block3_conv3'
        self.loss_model = Model(inputs=self.loss_model_vgg16.inputs, outputs=self.layers_vgg16)


    def _gram_matrix(self, x):
        """
        Calculation of Gram matrix.
        Static because it doesn't need to know the size of the output image.

        The Gram matrix can be efficiently calculated by
        reshaping x into a matrix of shape (C, (W*H)), where C = no. of channels,
        and then using the formula:
        G = matmul(x , x'), where x' is the transpose of x
        """

        # make it channel-first
        x = tf.transpose(x, (2, 0, 1))

        # flatten it
        # Note: -1 implies keep the total number of elements same
        # So the above line serves to flatten x
        # for example, a tensor of shape (1, 3, 4)
        # would be made into (1, 12)
        feature_vector = tf.reshape(x, (x.shape[0], -1))

        # Calculate the gram matrix: G = matmul(x, x')
        gram = tf.matmul(feature_vector, tf.transpose(feature_vector))

        # Return the normalizaed version of it
        channels = 3
        return (gram / (channels * self.img_h * self.img_w))


    def _style_loss(self, base, combo) -> float:
        """
        Calculates the style loss between two images using Gram matrix.
        """
        Gram_base = self._gram_matrix(base)
        Gram_combo = self._gram_matrix(combo)

        # Find the square of the Frobenius norm
        # of the normalized gram matrices G(combo) - G(base)
        return tf.reduce_sum(tf.square(Gram_base - Gram_combo))


    def _feature_recons_loss(self, base, combo) -> float:
        """
        Calculates the feature loss between the combined and base image.
        Can be made static, but didn't do so for the sake of uniformity.
        """
        # Normalize this too
        channels = 3
        return tf.reduce_sum(tf.square(combo - base)) / (channels * self.img_h * self.img_w)


    def _tv_loss(self, x) -> float:
        """
        Calculates total variation loss of tensor x.
        Obtained by shifting the image pixels by 1 pixel down
        And by 1 pixel right
        """
        # The tensor should be having rank of 4 here
        # 1 pixel shift down loss:
        a = tf.square(
            x[:, :self.img_h-1, :img_w-1, :] - x[:, 1:, :self.img_w-1, :]
        )
        # 1 pixel shift right loss:
        b = tf.square(
            x[:, :self.img_h-1, :img_w-1, :] - x[:, :self.img_h-1, 1:, :]
        )
        # combine a and b
        return tf.reduce_sum(tf.pow(a + b, 1.25))


    def _pixel_loss(y_pred, y_true) -> float:
        """
        Calculates simple pixel-wise loss.
        """
        c = 3  # channels
        h = self.img_h
        w = self.img_w
        return tf.reduce_sum(tf.square(y_true - y_pred) / (c * h * w))


    def compute_loss_3args(self, base, combo, styleref):
        # Send these 3 images together as a batch
        # Then we won't have to do 3 separate passes
        # Recall the 0th dimension of this input_tensor is the batch dimension
        print(base.shape, styleref.shape, combo.shape)
        print(type(base), type(styleref), type(combo))
        # Recall that currently, combo and base are actually unprocessed for VGG input
        # Also, combo and base are in the range [0, 1], but vgg16 expects them as [0, 255]
        base = base * 255
        combo = combo * 255
        base = vgg16.preprocess_input(base)
        combo = vgg16.preprocess_input(combo)
        
        input_tensor = tf.concat([base, styleref, combo], axis=0)
        # get the outputs of the vgg16
        outs = self.loss_model(input_tensor)

        # init loss
        loss = tf.zeros(shape=())

        # Add the content losses first
        # Note: I've used the words 'content loss' and 'feature reconstruction loss' interchangeably
        outs_content_feat = outs[self.feature_recons_layer]
        base_content = outs_content_feat[0,:,:,:]
        combo_content = outs_content_feat[2,:,:,:]
        loss += self.feature_recons_wt * self._feature_recons_loss(base_content, combo_content)

        # Now add the style reconstruction losses
        for style_layer in self.style_layers:
            # This is similar to calculating content loss
            # But we need this averaged over a number of layers

            outs_style_feat = outs[style_layer]
            styleref_style_content = outs_style_feat[1,:,:,:]
            combo_style_content = outs_style_feat[2,:,:,:]
            style_loss = self._style_loss(styleref_style_content, combo_style_content)
            n_sty_layers = len(self.style_layers)
            loss += self.style_wt * (style_loss / n_sty_layers)

        # Finally add the total variation loss and the pixel loss into the mix:
        loss += self.tv_wt * _tv_loss(combo)
        loss += self.pixel_wt * _pixel_loss(combo, base)

        return loss


    def get_loss_function(self, styleref):
        """
        Returns a (partial) function that accepts 2 arguments,
        as keras expects a loss function that takes 2 arguments while training.
        """
        # Recall that for style transfer, y_c = x_inp (base image)
        loss_funct = partial(self.compute_loss_3args, styleref=styleref)
        loss_funct.__name__ = 'perceptual_loss'
        
        return loss_funct