import os
import numpy as np
import cv2

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from attention import *

def stem_block(x, n_filter, strides):
    x_init = x

    ## Conv 1
    x = Conv2D(n_filter, (3, 3), padding="same", strides=strides)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(n_filter, (3, 3), padding="same")(x)

    ## Shortcut
    s  = Conv2D(n_filter, (1, 1), padding="same", strides=strides)(x_init)
    ## Add
    x = Add()([x, s])
    return x

def resnet_block(x, n_filter, strides=1):
    x_init = x

    ## Conv 1
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(n_filter, (3, 3), padding="same", strides=strides)(x)
    ## Conv 2
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(n_filter, (3, 3), padding="same", strides=1)(x)

    ## Shortcut
    s  = Conv2D(n_filter, (1, 1), padding="same", strides=strides)(x_init)
    ## Add
    x = Add()([x, s])
    return x

def attetion_block(g, x):
    """
        g: Output of Parallel Encoder block
        x: Output of Previous Decoder block
    """

    filters = x.shape[-1]

    g_conv = BatchNormalization()(g)
    g_conv = Activation("relu")(g_conv)
    g_conv = Conv2D(filters, (3, 3), padding="same")(g_conv)

    g_pool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(g_conv)

    x_conv = BatchNormalization()(x)
    x_conv = Activation("relu")(x_conv)
    x_conv = Conv2D(filters, (3, 3), padding="same")(x_conv)

    gc_sum = Add()([g_pool, x_conv])

    gc_conv = BatchNormalization()(gc_sum)
    gc_conv = Activation("relu")(gc_conv)
    gc_conv = Conv2D(filters, (3, 3), padding="same")(gc_conv)

    gc_mul = Multiply()([gc_conv, x])
    return gc_mul


def build_model(input_shape):
    inputs=Input(shape=input_shape)
    c0=inputs
    n_filters=[16,32,64,128,256]
    e1 = stem_block(c0, n_filters[0], strides=1)
    print('e1.shape: ', e1.shape)
    c1 = s_attention(e1)
    print('c1.shape: ', c1.shape)


    ## Encoder
    e2 = resnet_block(c1, n_filters[1], strides=2)
    c2 = s_attention(e2)
    e3 = resnet_block(e2, n_filters[2], strides=2)
    c3 = s_attention(e3)
    e4 = resnet_block(e3, n_filters[3], strides=2) 

    ## Bridge
    b1 = dual_attention(e4, filters=128)

    # Decoder
    d1 = attetion_block(e3, b1)
    d1 = UpSampling2D((2, 2))(d1)
    d1 = Concatenate()([d1, e3])
    print('d1.shape: ', d1.shape)
    d1 = resnet_block(d1, n_filters[3])


    d2 = attetion_block(e2, d1)
    d2 = UpSampling2D((2, 2))(d2)
    d2 = Concatenate()([d2, e2])
    d2 = resnet_block(d2, n_filters[2])   


    d3 = attetion_block(e1, d2)
    d3 = UpSampling2D((2, 2))(d3)
    d3 = Concatenate()([d3, e1])
    d3 = resnet_block(d3, n_filters[1])

    outputs = s_attention(d3)
    outputs = Conv2D(1, (1, 1), padding="same")(outputs)
    outputs = Activation("sigmoid")(outputs)

    ## Model
    model = Model(inputs, outputs)
    return model


if __name__ == "__main__":
    input_shape = (256, 256, 3)
    model = build_model(input_shape)
    model.summary()