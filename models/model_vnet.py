"""VNet architecture.

Diogo Amorim, 2018-07-10 V-Net implementation in Keras 2
https://arxiv.org/pdf/1606.04797.pdf
"""
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
import tensorflow as tf
from typing import Tuple, List

tf.compat.v1.disable_v2_behavior()


def BatchNormalization(name: str) -> tf.keras.layers.Layer:
    """Constructs a Batch Normalization layer.

    Args:
        name (str): Name of the layer.

    Returns:
        tf.keras.layers.Layer: Batch Normalization layer with fused \
        parameter set to False.
    """
    return tf.keras.layers.BatchNormalization(name=name, fused=False)


def Deconvolution3D(
    inputs: tf.Tensor,
    filters: int,
    kernel_size: Tuple[int],
    subsample: Tuple[int],
    name: str,
) -> tf.Tensor:
    """Constructs a 3D Deconvolution layer.

    Args:
        inputs: Input tensor.
        filters (int): Number of output filters.
        kernel_size (tuple[int]): Size of the kernel.
        subsample (tuple[int]): Strides for the deconvolution operation.
        name (str): Name of the layer.

    Returns:
        tf.Tensor: Output tensor after applying 3D Deconvolution.
    """

    strides = tuple(subsample)

    x = tf.keras.layers.Conv3DTranspose(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        activation=None,
        padding="same",
        data_format="channels_last",
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer=tf.zeros_initializer(),
        name=name,
        bias_regularizer=None,
    )(inputs)

    return x


def downward_layer(
    input_layer: tf.Tensor,
    n_convolutions: int,
    n_output_channels: int,
    number: int,
    strides: Tuple[int, int, int] = (2, 2, 2),
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Constructs a downward layer for a 3D convolutional network.

    Args:
        input_layer: Input tensor.
        n_convolutions (int): Number of convolutional layers.
        n_output_channels (int): Number of output channels.
        number (int): Layer number.
        strides (tuple[int]): Strides for the convolution operation.

    Returns:
        tuple[tf.Tensor, tf.Tensor]: Downsampled tensor and \
        skip connection tensor.
    """
    inl = input_layer

    for nnn in range(n_convolutions):
        inl = tf.keras.layers.Conv3D(
            filters=(n_output_channels // 2),
            kernel_size=5,
            padding="same",
            kernel_initializer="he_normal",
            name="conv_" + str(number) + "_" + str(nnn),
        )(inl)
        inl = BatchNormalization(name="batch_" + str(number) + "_" + str(nnn))(inl)
        inl = tf.keras.layers.ReLU(name="relu_" + str(number) + "_" + str(nnn))(inl)

    add_l = tf.math.add(inl, input_layer)
    downsample = tf.keras.layers.Conv3D(
        filters=n_output_channels,
        kernel_size=2,
        strides=strides,
        padding="same",
        kernel_initializer="he_normal",
        name="conv_" + str(number) + "_" + str(nnn + 1),
    )(add_l)
    downsample = BatchNormalization(name="batch_" + str(number) + "_" + str(nnn + 1))(
        downsample
    )
    downsample = tf.keras.layers.ReLU(name="relu_" + str(number) + "_" + str(nnn + 1))(
        downsample
    )
    return downsample, add_l


def upward_layer(
    input0: tf.Tensor,
    input1: tf.Tensor,
    n_convolutions: int,
    n_output_channels: int,
    number: int,
    strides: Tuple[int, int, int] = (2, 2, 2),
) -> tf.Tensor:
    """Constructs an upward layer for a 3D convolutional network.

    Args:
        input0: Input tensor for the main branch.
        input1: Input tensor for the skip connection.
        n_convolutions (int): Number of convolutional layers.
        n_output_channels (int): Number of output channels.
        number (int): Layer number.
        strides (tuple[int]): Strides for the deconvolution operation.

    Returns:
        tf.Tensor: Upsampled tensor.
    """
    merged = tf.concat([input0, input1], axis=4)
    inl = merged
    for nnn in range(n_convolutions):

        inl = tf.keras.layers.Conv3D(
            (n_output_channels * 4),
            kernel_size=5,
            padding="same",
            kernel_initializer="he_normal",
            name="conv_" + str(number) + "_" + str(nnn),
        )(inl)
        inl = BatchNormalization(name="batch_" + str(number) + "_" + str(nnn))(inl)
        inl = tf.keras.layers.ReLU(name="relu_" + str(number) + "_" + str(nnn))(inl)

    add_l = tf.math.add(inl, merged)
    shape = add_l.get_shape().as_list()
    new_shape = (
        1,
        shape[1] * 2,
        shape[2] * 2,
        shape[3] * 2,
        n_output_channels,
    )
    upsample = Deconvolution3D(
        add_l,
        n_output_channels,
        (2, 2, 2),
        subsample=strides,
        name="dconv_" + str(number) + "_" + str(nnn + 1),
    )
    upsample = BatchNormalization(name="batch_" + str(number) + "_" + str(nnn + 1))(
        upsample
    )
    return tf.keras.layers.ReLU(name="relu_" + str(number) + "_" + str(nnn + 1))(
        upsample
    )


def vnet(
    input_size: Tuple[int, int, int, int] = (128, 128, 128, 1),
    optimizer: tf.keras.optimizers.Optimizer = Adam(lr=1e-4),
    loss: str = "binary_crossentropy",
    metrics: List[str] = ["accuracy"],
) -> tf.keras.Model:
    """Constructs a 3D V-Net architecture for segmentation of Dedicated
    Ventilation or Gas exchange images.

    Args:
        input_size (tuple[int, int, int, int]): The size of the input images in\
        the format (height, width, depth, channels).
        optimizer (tf.keras.optimizers.Optimizer): The optimizer used for training\
        the model.
        loss (str): The loss function used for training the model.
        metrics (list[str]): List of metrics to be evaluated by the model during \
        training and testing.

    Returns:
        tf.keras.Model: A 3D V-Net model for segmentation.
    """
    # Layer 1
    input_gas = tf.keras.layers.Input(input_size)

    conv1 = tf.keras.layers.Conv3D(
        16,
        kernel_size=5,
        strides=1,
        padding="same",
        kernel_initializer="he_normal",
        name="conv_1",
    )(input_gas)
    conv1 = BatchNormalization(name="batch_1")(conv1)
    conv1 = tf.keras.layers.ReLU(name="relu_1")(conv1)
    repeat1 = tf.concat(16 * [input_gas], axis=-1)
    add1 = tf.math.add(conv1, repeat1)
    down1 = tf.keras.layers.Conv3D(
        32,
        kernel_size=(2, 2, 2),
        strides=2,
        padding="same",
        kernel_initializer="he_normal",
        name="down_1",
    )(add1)
    down1 = BatchNormalization(name="batch_1_2")(down1)
    down1 = tf.keras.layers.ReLU(name="relu_1_2")(down1)

    # Layer 2,3,4
    down2, add2 = downward_layer(down1, 2, 64, 2)
    down3, add3 = downward_layer(down2, 3, 128, 3)
    down4, add4 = downward_layer(down3, 3, 256, 4)

    # Layer 5
    conv_5_1 = tf.keras.layers.Conv3D(
        256,
        kernel_size=(5, 5, 5),
        padding="same",
        kernel_initializer="he_normal",
        name="conv_5_1",
    )(down4)
    conv_5_1 = BatchNormalization(name="batch_5_1")(conv_5_1)
    conv_5_1 = tf.keras.layers.ReLU(name="relu_5_1")(conv_5_1)
    conv_5_2 = tf.keras.layers.Conv3D(
        256,
        kernel_size=(5, 5, 5),
        padding="same",
        kernel_initializer="he_normal",
        name="conv_5_2",
    )(conv_5_1)
    conv_5_2 = BatchNormalization(name="batch_5_2")(conv_5_2)
    conv_5_2 = tf.keras.layers.ReLU(name="relu_5_2")(conv_5_2)
    conv_5_3 = tf.keras.layers.Conv3D(
        256,
        kernel_size=(5, 5, 5),
        padding="same",
        kernel_initializer="he_normal",
        name="conv_5_3",
    )(conv_5_2)
    conv_5_3 = BatchNormalization(name="batch_5_3")(conv_5_3)
    conv_5_3 = tf.keras.layers.ReLU(name="relu_5_3")(conv_5_3)
    add5 = tf.math.add(conv_5_3, down4)

    aux_shape = add5.get_shape()
    upsample_5 = Deconvolution3D(
        add5, 128, (2, 2, 2), subsample=(2, 2, 2), name="dconv_5"
    )

    upsample_5 = BatchNormalization(name="batch_5_4")(upsample_5)
    upsample_5 = tf.keras.layers.ReLU(name="relu_5_4")(upsample_5)

    # Layer 6,7,8
    upsample_6 = upward_layer(upsample_5, add4, 3, 64, 6)
    upsample_7 = upward_layer(upsample_6, add3, 3, 32, 7)
    upsample_8 = upward_layer(upsample_7, add2, 2, 16, 8)

    # Layer 9
    merged_9 = tf.concat([upsample_8, add1], axis=4)
    conv_9_1 = tf.keras.layers.Conv3D(
        32,
        kernel_size=(5, 5, 5),
        padding="same",
        kernel_initializer="he_normal",
        name="conv_9_1",
    )(merged_9)
    conv_9_1 = BatchNormalization(name="batch_9_1")(conv_9_1)
    conv_9_1 = tf.keras.layers.ReLU(name="relu_9_1")(conv_9_1)
    add_9 = tf.math.add(conv_9_1, merged_9)
    conv_9_2 = tf.keras.layers.Conv3D(
        1,
        kernel_size=(1, 1, 1),
        padding="same",
        kernel_initializer="he_normal",
        name="conv_9_2",
    )(add_9)
    conv_9_2 = BatchNormalization(name="batch_9_2")(conv_9_2)
    conv_9_2 = tf.keras.layers.ReLU(name="relu_9_2")(conv_9_2)

    sigmoid_v = tf.keras.layers.Conv3D(
        1,
        kernel_size=(1, 1, 1),
        padding="same",
        kernel_initializer="he_normal",
        name="conv_sigm_1",
    )(conv_9_2)
    sigmoid_v = BatchNormalization(name="batch_sigm_1")(sigmoid_v)
    sigmoid_v = tf.keras.layers.Activation(activation="sigmoid")(sigmoid_v)

    model = Model(inputs=input_gas, outputs=sigmoid_v)

    return model


def vnet_2dgre(
    input_size: Tuple[int, int, int, int] = (128, 128, 14, 1),
    optimizer: tf.keras.optimizers.Optimizer = Adam(lr=1e-4),
    loss: str = "binary_crossentropy",
    metrics: List[str] = ["accuracy"],
) -> tf.keras.Model:
    """Constructs a 2.5D V-Net architecture for segmentation of 2D Gradient-
    Recalled Echo (GRE) images.

    Args:
        input_size (tuple[int, int, int, int]): The size of the input images in\
        the format (height, width, depth, channels).
        optimizer (tf.keras.optimizers.Optimizer): The optimizer used for training\
        the model.
        loss (str): The loss function used for training the model.
        metrics (list[str]): List of metrics to be evaluated by the model during\
        training and testing.

    Returns:
        tf.keras.Model: A 2.5D V-Net model for segmentation.
    """
    # Layer 1
    input_gas = tf.keras.layers.Input(input_size)

    conv1 = tf.keras.layers.Conv3D(
        16,
        kernel_size=5,
        strides=1,
        padding="same",
        kernel_initializer="he_normal",
        name="conv_1",
    )(input_gas)
    conv1 = BatchNormalization(name="batch_1")(conv1)
    conv1 = tf.keras.layers.ReLU(name="relu_1")(conv1)
    repeat1 = tf.concat(16 * [input_gas], axis=-1)
    add1 = tf.math.add(conv1, repeat1)
    down1 = tf.keras.layers.Conv3D(
        32,
        kernel_size=(2, 2, 2),
        strides=(2, 2, 1),
        padding="same",
        kernel_initializer="he_normal",
        name="down_1",
    )(add1)
    down1 = BatchNormalization(name="batch_1_2")(down1)
    down1 = tf.keras.layers.ReLU(name="relu_1_2")(down1)

    # Layer 2,3,4
    down2, add2 = downward_layer(down1, 2, 64, 2, (2, 2, 1))
    down3, add3 = downward_layer(down2, 3, 128, 3, (2, 2, 1))
    down4, add4 = downward_layer(down3, 3, 256, 4, (2, 2, 1))

    # Layer 5
    conv_5_1 = tf.keras.layers.Conv3D(
        256,
        kernel_size=(5, 5, 5),
        padding="same",
        kernel_initializer="he_normal",
        name="conv_5_1",
    )(down4)
    conv_5_1 = BatchNormalization(name="batch_5_1")(conv_5_1)
    conv_5_1 = tf.keras.layers.ReLU(name="relu_5_1")(conv_5_1)
    conv_5_2 = tf.keras.layers.Conv3D(
        256,
        kernel_size=(5, 5, 5),
        padding="same",
        kernel_initializer="he_normal",
        name="conv_5_2",
    )(conv_5_1)
    conv_5_2 = BatchNormalization(name="batch_5_2")(conv_5_2)
    conv_5_2 = tf.keras.layers.ReLU(name="relu_5_2")(conv_5_2)
    conv_5_3 = tf.keras.layers.Conv3D(
        256,
        kernel_size=(5, 5, 5),
        padding="same",
        kernel_initializer="he_normal",
        name="conv_5_3",
    )(conv_5_2)
    conv_5_3 = BatchNormalization(name="batch_5_3")(conv_5_3)
    conv_5_3 = tf.keras.layers.ReLU(name="relu_5_3")(conv_5_3)
    add5 = tf.math.add(conv_5_3, down4)

    aux_shape = add5.get_shape()
    upsample_5 = Deconvolution3D(
        add5, 128, (2, 2, 2), subsample=(2, 2, 1), name="dconv_5"
    )

    upsample_5 = BatchNormalization(name="batch_5_4")(upsample_5)
    upsample_5 = tf.keras.layers.ReLU(name="relu_5_4")(upsample_5)

    # Layer 6,7,8
    upsample_6 = upward_layer(upsample_5, add4, 3, 64, 6, (2, 2, 1))
    upsample_7 = upward_layer(upsample_6, add3, 3, 32, 7, (2, 2, 1))
    upsample_8 = upward_layer(upsample_7, add2, 2, 16, 8, (2, 2, 1))

    # Layer 9
    merged_9 = tf.concat([upsample_8, add1], axis=4)
    conv_9_1 = tf.keras.layers.Conv3D(
        32,
        kernel_size=(5, 5, 5),
        padding="same",
        kernel_initializer="he_normal",
        name="conv_9_1",
    )(merged_9)
    conv_9_1 = BatchNormalization(name="batch_9_1")(conv_9_1)
    conv_9_1 = tf.keras.layers.ReLU(name="relu_9_1")(conv_9_1)
    add_9 = tf.math.add(conv_9_1, merged_9)
    conv_9_2 = tf.keras.layers.Conv3D(
        1,
        kernel_size=(1, 1, 1),
        padding="same",
        kernel_initializer="he_normal",
        name="conv_9_2",
    )(add_9)
    conv_9_2 = BatchNormalization(name="batch_9_2")(conv_9_2)
    conv_9_2 = tf.keras.layers.ReLU(name="relu_9_2")(conv_9_2)

    # softmax = Softmax()(conv_9_2)
    sigmoid_v = tf.keras.layers.Conv3D(
        1,
        kernel_size=(1, 1, 1),
        padding="same",
        kernel_initializer="he_normal",
        name="conv_sigm_1",
    )(conv_9_2)
    sigmoid_v = BatchNormalization(name="batch_sigm_1")(sigmoid_v)
    sigmoid_v = tf.keras.layers.Activation(activation="sigmoid")(sigmoid_v)

    model = Model(inputs=input_gas, outputs=sigmoid_v)

    return model
