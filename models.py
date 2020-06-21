# Neural Style transfer
# Johnson et al. method (built on the method by Gatys et al.)

# This file defines the Image Transformation Network

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Activation, Add, Input
from tensorflow.keras.layers import UpSampling2D, Conv2DTranspose
from tensorflow.keras.layers import InputSpec, BatchNormalization
from keras_contrib.normalization.instancenormalization import InstanceNormalization

from padding import ReflectionPadding2D
from blocks import ResidualBlock, DownsamplingBlock, UpsamplingBlock
import pickle


# The Image Transformation Network
class ImageTransformNetwork():
    """
    Creates the image transformation network, with the architecture:
    c9s1-32,d64,d128,R128,R128,R128,R128,R128,u64,u32,c9s1-3
    (as suggested in the paper)
    """

    def __init__(self):
        # TODO: Complete the initialization
        self._build_network()


    def _residual_bock(self, input_layer, n_filters):
        short_circuit = input_layer
        x = Conv2D(filters=n_filters, kernel_size=(3, 3), strides=1, padding='same')(input_layer)
        x = InstanceNormalization(axis=-1, center=False, scale=False)(x)
        x = Activation('relu')(x)
        
        x = Conv2D(filters=n_filters, kernel_sizw=(3, 3), strides=1, padding='same')(x)
        x = InstanceNormalization(axis=-1, center=False, scale=False)(x)

        x = Add()([short_circuit, x])
        return x


    # For the input:
    def _conv_inp_c9s1f32(self, input_tensor):
        x = ReflectionPadding2D(input_tensor)
        x = Conv2D(filters=32, strides=1, kernel_size=(9, 9), padding='valid')(x)
        return x

    # For the output
    def _conv_out_c9s1f3(self, input_tensor):
        x = Conv2D(filters=3, strides=1, kernel_size=(9, 9), padding='same')(input_tensor)
        return x


    def _build_network(self, n_residual_blocks=5):
        inp = Input(shape=(None, None, None, 3), name='input_layer')
        x = self._conv_inp_c9s1f32(inp)
        # Downsample:
        x = DownsamplingBlock(filters=64)(x)
        x = DownsamplingBlock(filters=128)(x)
        
        # Stack up residual blocks:
        for i in range(n_residual_blocks):
            x = ResidualBlock(filters=128)(x)
        
        # Get the output
        x = UpsamplingBlock(filters=64)(x)
        x = UpsamplingBlock(filters=32)(x)
        out = _conv_out_c9s1f3(x)

        self.model = Model(inputs=inp, outputs=out)

    
    def get_model(self):
        return self.model