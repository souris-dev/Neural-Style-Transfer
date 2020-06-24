# Blocks out of which the ImageTransformNetwork is composed of

from tensorflow.keras.layers import Conv2D, Activation, Add
from tensorflow.keras.layers import UpSampling2D, Conv2DTranspose
from tensorflow.keras.layers import InputSpec, BatchNormalization
from tensorflow.keras.layers import Layer
from tensorflow_addons.layers.normalizations import InstanceNormalization


# Residual blocks
class ResidualBlock(Layer):
    def __init__(self, filters, norm='instancenorm', **kwargs):
        self.filters = filters
        self.input_spec = [InputSpec(ndim=4)]
        super(ResidualBlock, self).__init__(**kwargs)

        self.conv_1 = Conv2D(filters=self.filters, kernel_size=(3, 3), strides=1, padding='same')

        if norm == 'instancenorm':
            self.norm_1 = InstanceNormalization(axis=-1, center=False, scale=False)
            self.norm_2 = InstanceNormalization(axis=-1, center=False, scale=False)
        elif norm == 'batchnorm':
            self.norm_1 = BatchNormalization()
            self.norm_2 = BatchNormalization()
        
        self.acti_1 = Activation('relu')
        self.conv_2 = Conv2D(filters=self.filters, kernel_size=(3, 3), strides=1, padding='same')
        self.adder = Add()

    def call(self, inputs):
        short_circuit = inputs
        layer = self.conv_1(inputs)
        layer = self.norm_1(layer)
        layer = self.acti_1(layer)
        
        layer = self.conv_2(layer)
        layer = self.norm_2(layer)
        out = self.adder([short_circuit, layer])

        return out

     # To make the block (layer) serializable
    def get_config(self):
        config = super(ResidualBlock, self).get_config()
        config.update({"filters": self.filters, "norm_type": self.norm_type})
        return config


# Downsampling blocks
class DownsamplingBlock(Layer):
    def __init__(self, filters, norm='instancenorm', **kwargs):
        self.filters = filters
        self.norm_type = norm
        self.input_spec = [InputSpec(ndim=4)]
        super(DownsamplingBlock, self).__init__(**kwargs)

        self.conv_1 = Conv2D(filters=self.filters, kernel_size=(3, 3), strides=2, padding='same')
        if norm == 'instancenorm':
            self.norm_layer = InstanceNormalization(axis=-1, center=False, scale=False)
        elif norm == 'batchnorm':
            self.norm_layer = BatchNormalization()
        self.acti = Activation('relu')
        
    def call(self, inputs):
        layer = self.conv_1(inputs)
        layer = self.norm_layer(layer)
        layer = self.acti(layer)

        return layer

    # To make the block (layer) serializable
    def get_config(self):
        config = super(DownsamplingBlock, self).get_config()
        config.update({"filters": self.filters, "norm_type": self.norm_type})
        return config


# Upsampling blocks
class UpsamplingBlock(Layer):
    def __init__(self, filters, norm='instancenorm', use_conv_trans=True, **kwargs):
        """
        use_conv_trans, if set to False, will make this class
        use UpSampling2D instead.
        This argument controls the layer used for upsampling.
        """
        self.filters = filters
        self.use_conv_trans = use_conv_trans
        self.norm_type = norm
        self.input_spec = [InputSpec(ndim=4)]
        super(UpsamplingBlock, self).__init__(**kwargs)

        if (use_conv_trans):
            self.convtr_1 = Conv2DTranspose(filters=self.filters, kernel_size=(3, 3), strides=2, padding='same')
        else:
            self.upsampling_1 = UpSampling2D()
            self.conv_1 = Conv2D(filters=self.filters, kernel_size=3, strides=1, padding='same')

        if self.norm_type == 'instancenorm':
            self.norm_1 = InstanceNormalization(axis=-1, center=False, scale=False)
        elif self.norm_type == 'batchnorm':
            self.norm_1 = BatchNormalization()

        self.activ = Activation('relu')

    def call(self, inputs):
        if (self.use_conv_trans):
            layer = self.convtr_1(inputs)
        else:
            layer = self.upsampling_1(inputs)
            layer = self.conv_1(layer)
        
        layer = self.norm_1(layer)
        layer = self.activ(layer)

        return layer

    # To make the block (layer) serializable
    def get_config(self):
        config = super(UpsamplingBlock, self).get_config()
        config.update({"filters": self.filters, "norm_type": self.norm_type, "use_conv_trans": self.use_conv_trans})
        return config