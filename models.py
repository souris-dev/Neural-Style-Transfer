# Neural Style transfer
# Johnson et al. method (built on the method by Gatys et al.)

# This file defines the Image Transformation Network

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Activation, Add, Input
from tensorflow.keras.layers import UpSampling2D, Conv2DTranspose
from tensorflow.keras.layers import InputSpec, BatchNormalization

from tensorflow_addons.layers.normalizations import InstanceNormalization
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import numpy as np

from padding import ReflectionPadding2D
from blocks import ResidualBlock, DownsamplingBlock, UpsamplingBlock
from losses import LossGenerator
import pickle


# The Image Transformation Network
class ImageTransformNetwork():
    """
    Creates the image transformation network, with the architecture:
    c9s1-32,d64,d128,R128,R128,R128,R128,R128,u64,u32,c9s1-3
    (as suggested in the paper)
    """

    def __init__(self, style_img: np.ndarray):
        self.styleref = style_img
        self._build_network()


    @classmethod
    def from_keras_saved(classname, path, weights_path=False):
        """
        Factory function to create an ImageTransformNetwork
        from a saved file.
        """
        img_tr_net = classname()
        img_tr_net.model = load_model(path)
        return img_tr_net


    @classmethod
    def just_want_the_model(classname):
        """
        Gives only the model after creating an ImageTransformNetwork
        """
        img_tr_net = classname()
        return img_tr_net.model


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
        # padding = (n_filters - 1) / 2
        pad = 4
        x = ReflectionPadding2D(padding=(p, p))(input_tensor)
        x = Conv2D(filters=32, strides=1, kernel_size=(9, 9), padding='valid')(x)
        return x

    # For the output
    def _conv_out_c9s1f3(self, input_tensor):
        x = Conv2D(filters=3, strides=1, kernel_size=(9, 9), padding='same')(input_tensor)
        return x


    def _build_network(self, n_residual_blocks=5):
        inp = Input(shape=(None, None, 3), name='input_layer')
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
        out = self._conv_out_c9s1f3(x)

        self.model = Model(inputs=inp, outputs=out)


    def compile_model(self, loss_measurer: LossGenerator):
        loss_fn = loss_measurer.get_loss_function(self.styleref)
        optimizer = Adam(lr=0.001)
        self.model.compile(loss=loss_fn, optimizer=optimizer)


    def train_model(self, generator, epochs, model_checkpoint_prefix: str):
        model_chkp = ModelCheckpoint(filepath=model_checkpoint_prefix + '.h5')
        callbacks = [model_chkp]
        self.model.fit(generator, epochs=epochs, callbacks=callbacks)

    # Some util functions    
    def get_model(self):
        return self.model

    def save_model(self, path):
        self.model.save(path)

    def save_model_weights(self, path):
        self.model.save_weights(path)

    def load_model(self, path):
        self.model = load_model(path)

    
    # To save the model as .pb for use in OpenCV
    def save_model_pb(self, path):
        # Convert Keras model to ConcreteFunction
        full_model = tf.function(lambda x: model(x))
        full_model = full_model.get_concrete_function(
            tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

        # Get frozen ConcreteFunction
        frozen_func = convert_variables_to_constants_v2(full_model)
        frozen_func.graph.as_graph_def()

        layers = [op.name for op in frozen_func.graph.get_operations()]
        print("-" * 50)
        print("Frozen model layers: ")
        for layer in layers:
            print(layer)

        print("-" * 50)
        print("Frozen model inputs: ")
        print(frozen_func.inputs)
        print("Frozen model outputs: ")
        print(frozen_func.outputs)

        # Save frozen graph from frozen ConcreteFunction to hard drive
        tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                        logdir="./frozen_models",
                        name=path,
                        as_text=False)
