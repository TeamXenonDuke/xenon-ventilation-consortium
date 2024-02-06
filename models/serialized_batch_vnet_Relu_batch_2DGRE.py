
"""
Diogo Amorim, 2018-07-10
V-Net implementation in Keras 2
https://arxiv.org/pdf/1606.04797.pdf
"""

import functools

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.keras.layers import *
from tensorflow.keras import activations
#from tensorflow.keras.layers.advanced_activations import ReLU
from  tensorflow.keras.layers import ReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.layers import BatchNormalization 
import tensorflow.keras as keras 
from tensorflow.keras.initializers import GlorotUniform


def Deconvolution3D(inputs, filters, kernel_size, subsample,name):

    strides = tuple(subsample) 

    x = keras.layers.Conv3DTranspose(
                                    filters=filters,
                                    kernel_size=kernel_size,
                                    strides=strides,
                                    activation=None,
                                    padding='same',
                                    data_format='channels_last',
                                    use_bias=True,
                                    kernel_initializer=tf.glorot_uniform_initializer(),
                                    bias_initializer=tf.zeros_initializer(),
                                    name = name,
                                    bias_regularizer=None)(inputs)

    return x
"""

def get_config(self):
    config = {'output_shape_':self.output_shape_,
              'strides':self.strides
              }

    if hasattr(self, 'filters'):
        config['filters'] = self.filters
    if hasattr(self, 'kernel_size'):
        config['kernel_size'] = self.kernel_size

    base_config = super(Deconvolution3D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


def get_config(self):
    return {'filters': self.filters,
              'kernel_size': self.kernel_size,
              'output_shape_':self.output_shape_,
              'strides':self.strides}


def get_config(self):
    config = {'layer': {'class_name': self.layer.__class__.__name__,
                        'config': self.layer.get_config()}}
    base_config = super(Deconvolution3D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

@classmethod
def from_config(cls, config, custom_objects=None):
    from . import deserialize as deserialize_layer
    layer = deserialize_layer(config.pop('layer'),
                              custom_objects=custom_objects)
    return cls(layer, **config)
"""

def downward_layer(input_layer, n_convolutions, n_output_channels,number):
    inl = input_layer

    for nnn in range(n_convolutions):
        inl = Conv3D(filters=(n_output_channels // 2), kernel_size=5,padding='same', kernel_initializer='he_normal',name = "conv_"+str(number)+"_"+str(nnn))(inl)
        inl = BatchNormalization(name = "batch_"+str(number)+"_"+str(nnn))(inl)
        inl = ReLU(name = "relu_"+str(number)+"_"+str(nnn))(inl)
        
    add_l = add([inl, input_layer])
    downsample = Conv3D(filters=n_output_channels, kernel_size=2, strides=(2,2,1),
                        padding='same', kernel_initializer='he_normal',name = "conv_"+str(number)+"_"+str(nnn+1))(add_l)
    downsample = BatchNormalization(name = "batch_"+str(number)+"_"+str(nnn+1))(downsample)
    downsample = ReLU(name = "relu_"+str(number)+"_"+str(nnn+1))(downsample)
    return downsample, add_l


def upward_layer(input0, input1, n_convolutions, n_output_channels,number):
    merged = concatenate([input0, input1], axis=4)
    inl = merged
    for nnn in range(n_convolutions):
  
        inl = Conv3D((n_output_channels * 4), kernel_size=5,padding='same', kernel_initializer='he_normal',name = "conv_"+str(number)+"_"+str(nnn))(inl)
        inl= BatchNormalization(name = "batch_"+str(number)+"_"+str(nnn))(inl)
        inl= ReLU(name = "relu_"+str(number)+"_"+str(nnn))(inl)
        
    add_l = add([inl, merged])
    shape = add_l.get_shape().as_list()
    new_shape = (1, shape[1] * 2, shape[2] * 2, shape[3] * 1, n_output_channels)
    upsample = Deconvolution3D(add_l,n_output_channels, (2, 2, 1), subsample=(2, 2, 1),name = "dconv_"+str(number)+"_"+str(nnn+1))
    upsample = BatchNormalization(name = "batch_"+str(number)+"_"+str(nnn+1))(upsample)
    return ReLU(name = "relu_"+str(number)+"_"+str(nnn+1))(upsample)

def vnet_saved(input_size=(128, 128, 128, 1), optimizer=Adam(lr=1e-4),
         loss='binary_crossentropy', metrics=['accuracy']):
         # loss='categorical_crossentropy', metrics=['categorical_accuracy']):
    # Layer 1
    inputs = Input(input_size)
    conv1 = Conv3D(16, kernel_size=5, strides=1, padding='same', kernel_initializer='he_normal',name="conv_1")(inputs)
    conv1 = BatchNormalization(name="batch_1")(conv1)
    conv1 = ReLU(name = "relu_1")(conv1)
    repeat1 = concatenate(16 * [inputs], axis=-1)
    add1 = add([conv1, repeat1])
    down1 = Conv3D(32, kernel_size=(2,2,2), strides=2, padding='same', kernel_initializer='he_normal',name = "down_1")(add1)
    down1 = BatchNormalization(name="batch_1_2")(down1)
    down1 = ReLU(name = "relu_1_2")(down1)

    # Layer 2,3,4
    down2, add2 = downward_layer(down1, 2, 64, 2)
    down3, add3 = downward_layer(down2, 3, 128, 3)
    down4, add4 = downward_layer(down3, 3, 256, 4)

    # Layer 5
    # !Mudar kernel_size=(5, 5, 5) quando imagem > 64!
    conv_5_1 = Conv3D(256, kernel_size=(5, 5, 5), padding='same', kernel_initializer='he_normal',name="conv_5_1")(down4)
    conv_5_1 = BatchNormalization(name="batch_5_1")(conv_5_1)
    conv_5_1 = ReLU(name = "relu_5_1")(conv_5_1)
    conv_5_2 = Conv3D(256, kernel_size=(5, 5, 5), padding='same', kernel_initializer='he_normal',name="conv_5_2")(conv_5_1)
    conv_5_2 = BatchNormalization(name="batch_5_2")(conv_5_2)
    conv_5_2 = ReLU(name = "relu_5_2")(conv_5_2)
    conv_5_3 = Conv3D(256, kernel_size=(5, 5, 5), padding='same', kernel_initializer='he_normal',name="conv_5_3")(conv_5_2)
    conv_5_3 = BatchNormalization(name="batch_5_3")(conv_5_3)
    conv_5_3 = ReLU(name = "relu_5_3")(conv_5_3)
    add5 = add([conv_5_3, down4])
    aux_shape = add5.get_shape()
    upsample_5 = Deconvolution3D(add5,128, (2, 2, 2), subsample=(2, 2, 2),name = "dconv_5")

    upsample_5 = BatchNormalization(name="batch_5_4")(upsample_5)
    upsample_5 = ReLU(name = "relu_5_4")(upsample_5)

    # Layer 6,7,8
    upsample_6 = upward_layer(upsample_5, add4, 3, 64,6)
    upsample_7 = upward_layer(upsample_6, add3, 3, 32,7)
    upsample_8 = upward_layer(upsample_7, add2, 2, 16,8)

    # Layer 9
    merged_9 = concatenate([upsample_8, add1], axis=4)
    conv_9_1 = Conv3D(32, kernel_size=(5, 5, 5), padding='same', kernel_initializer='he_normal',name="conv_9_1")(merged_9)
    conv_9_1 = BatchNormalization(name="batch_9_1")(conv_9_1)
    conv_9_1 = ReLU(name = "relu_9_1")(conv_9_1)
    add_9 = add([conv_9_1, merged_9])
    # conv_9_2 = Conv3D(1, kernel_size=(1, 1, 1), padding='same', kernel_initializer='he_normal')(add_9)
    conv_9_2 = Conv3D(1, kernel_size=(1, 1, 1), padding='same', kernel_initializer='he_normal',name="conv_9_2")(add_9)
    conv_9_2 = BatchNormalization(name="batch_9_2")(conv_9_2)
    conv_9_2 = ReLU(name = "relu_9_2")(conv_9_2)(2,2,1)
    sigmoid_v = Conv3D(1, kernel_size=(1, 1, 1), padding='same', kernel_initializer='he_normal',name="conv_sigm_1")(conv_9_2)
    sigmoid_v  = BatchNormalization(name="batch_sigm_1")(sigmoid_v)
    sigmoid_v = keras.layers.Activation(activation='sigmoid')(sigmoid_v)

    model = Model(inputs=inputs, outputs=sigmoid_v)
    # model = Model(inputs=inputs, outputs=softmax)

    return model


def vnet(input_size=(128, 128, 14, 1), optimizer=Adam(lr=1e-4),
         loss='binary_crossentropy', metrics=['accuracy']):
         # loss='categorical_crossentropy', metrics=['categorical_accuracy']):
    # Layer 1
    input_gas = Input(input_size)


    conv1 = Conv3D(16, kernel_size=5, strides=1, padding='same', kernel_initializer='he_normal',name="conv_1")(input_gas)
    conv1 = BatchNormalization(name="batch_1")(conv1)
    conv1 = ReLU(name = "relu_1")(conv1)
    repeat1 = concatenate(16 * [input_gas], axis=-1)
    add1 = add([conv1, repeat1])
    down1 = Conv3D(32, kernel_size=(2,2,2), strides=(2,2,1), padding='same', kernel_initializer='he_normal',name = "down_1")(add1)
    down1 = BatchNormalization(name="batch_1_2")(down1)
    down1 = ReLU(name = "relu_1_2")(down1)

    # Layer 2,3,4
    down2, add2 = downward_layer(down1, 2, 64, 2)
    down3, add3 = downward_layer(down2, 3, 128, 3)
    down4, add4 = downward_layer(down3, 3, 256, 4)

    # Layer 5
    # !Mudar kernel_size=(5, 5, 5) quando imagem > 64!
    conv_5_1 = Conv3D(256, kernel_size=(5, 5, 5), padding='same', kernel_initializer='he_normal',name="conv_5_1")(down4)
    conv_5_1 = BatchNormalization(name="batch_5_1")(conv_5_1)
    conv_5_1 = ReLU(name = "relu_5_1")(conv_5_1)
    conv_5_2 = Conv3D(256, kernel_size=(5, 5, 5), padding='same', kernel_initializer='he_normal',name="conv_5_2")(conv_5_1)
    conv_5_2 = BatchNormalization(name="batch_5_2")(conv_5_2)
    conv_5_2 = ReLU(name = "relu_5_2")(conv_5_2)
    conv_5_3 = Conv3D(256, kernel_size=(5, 5, 5), padding='same', kernel_initializer='he_normal',name="conv_5_3")(conv_5_2)
    conv_5_3 = BatchNormalization(name="batch_5_3")(conv_5_3)
    conv_5_3 = ReLU(name = "relu_5_3")(conv_5_3)
    add5 = add([conv_5_3, down4])

    aux_shape = add5.get_shape()
    upsample_5 = Deconvolution3D(add5,128, (2, 2, 1), subsample=(2, 2, 1),name = "dconv_5")

    upsample_5 = BatchNormalization(name="batch_5_4")(upsample_5)
    upsample_5 = ReLU(name = "relu_5_4")(upsample_5)

    # Layer 6,7,8
    upsample_6 = upward_layer(upsample_5, add4, 3, 64,6)
    upsample_7 = upward_layer(upsample_6, add3, 3, 32,7)
    upsample_8 = upward_layer(upsample_7, add2, 2, 16,8)

    # Layer 9
    merged_9 = concatenate([upsample_8, add1], axis=4)
    conv_9_1 = Conv3D(32, kernel_size=(5, 5, 5), padding='same', kernel_initializer='he_normal',name="conv_9_1")(merged_9)
    conv_9_1 = BatchNormalization(name="batch_9_1")(conv_9_1)
    conv_9_1 = ReLU(name = "relu_9_1")(conv_9_1)
    add_9 = add([conv_9_1, merged_9])
    # conv_9_2 = Conv3D(1, kernel_size=(1, 1, 1), padding='same', kernel_initializer='he_normal')(add_9)
    conv_9_2 = Conv3D(1, kernel_size=(1, 1, 1), padding='same', kernel_initializer='he_normal',name="conv_9_2")(add_9)
    conv_9_2 = BatchNormalization(name="batch_9_2")(conv_9_2)
    conv_9_2 = ReLU(name = "relu_9_2")(conv_9_2)

    # softmax = Softmax()(conv_9_2)
    sigmoid_v = Conv3D(1, kernel_size=(1, 1, 1), padding='same', kernel_initializer='he_normal',name="conv_sigm_1")(conv_9_2)
    sigmoid_v  = BatchNormalization(name="batch_sigm_1")(sigmoid_v)
    sigmoid_v = keras.layers.Activation(activation='sigmoid')(sigmoid_v)

    model = Model(inputs=input_gas, outputs=sigmoid_v)
    # model = Model(inputs=inputs, outputs=softmax)
    
    return model


#model.load_weights("./model_Best_vnetAnatomy_Sakib_epoch120.h5")
#model.save_weights("./model_Best_vnetAnatomy_Sakib_epoch120_name.h5")

#model = vnet()
#model.load_weights("/home/peppermint/Desktop/CIVM/simpleVNet/VNet/weights_vnet.h5")
# model.summary(line_length=133)
