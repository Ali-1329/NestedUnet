import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from attention import *
from tensorflow.keras.utils import get_custom_objects
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def attention_block(g, x):
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


class GroupedConv2D(object):

# channel haro split mikone be tedade group ha va har group Conv mizane va natije ro baham concat mikone


    def __init__(self, filters, kernel_size, use_keras=True, **kwargs):
        # kernel_size ye liste = [3, 3, 3, 3, ....]
        # self._groups = masalan 3, chon ye adade 
      
        self._groups = len(kernel_size)
        self._channel_axis = -1

        self._convs = []
        splits = self._split_channels(filters, self._groups)
        for i in range(self._groups):
            self._convs.append(self._get_conv2d(splits[i], kernel_size[i], use_keras, **kwargs))

    def _get_conv2d(self, filters, kernel_size, use_keras, **kwargs):
        """A helper function to create Conv2D layer."""
        if use_keras:
            return Conv2D(filters=filters, kernel_size=kernel_size, **kwargs)
        else:
            return Conv2D(filters=filters, kernel_size=kernel_size, **kwargs)

    def _split_channels(self, total_filters, num_groups):
        split = [total_filters // num_groups for _ in range(num_groups)]    
         # masalan:  split = [14, 14, 14, 14] yani tedade channel haro be tedade groups taghsim mikonim
         # tedade channel hayi ke be har group mirese ro moshakhas mikone
        split[0] += total_filters - sum(split)
        return split

    def __call__(self, inputs):
        if len(self._convs) == 1:
            return self._convs[0](inputs)


        filters = inputs.shape[self._channel_axis]
        splits = self._split_channels(filters, len(self._convs))
        x_splits = tf.split(inputs, splits, self._channel_axis)
        # tf.split(tensor, [3, 3, 3, 3, 3], axis=-1)     yani dar rastaye channels feature_map ro tagsim mikone b3 5 ta gesmat ke harkodum 3 ta channel daran
        x_outputs = [c(x) for x, c in zip(x_splits, self._convs)]    # yani be har split Convolution mizanim va natije ro to ye list mizarim
        # [print(xx.shape) for xx in x_outputs]
        x = tf.concat(x_outputs, self._channel_axis)
        # hameye inaro ke natijeye Convolution roye split haye mokhtalef hastan ro baham Concat mikonim
        return x


# to 2 ta Conv avali tedade filter ha, stem_width ta ast, to Conv sevomi, stem_width*2 ta ast
def make_stem(input_tensor, stem_width=64, active='relu'):      # stem_width: tedade filter hast
    x = input_tensor
    channel_axis = -1
    # inja strides=2 hast va spatial_dimensions nesf mishe
    x = Conv2D(stem_width, kernel_size=3, strides=1, padding="same", kernel_initializer="he_normal",
                use_bias=False, data_format="channels_last")(x)

    x = BatchNormalization(axis=channel_axis, epsilon=1.001e-5)(x)
    x = Activation(active)(x)

    x = Conv2D(stem_width, kernel_size=3, strides=1, padding="same",
                kernel_initializer="he_normal", use_bias=False, data_format="channels_last")(x)

    # Shortcut
    shortcut = Conv2D(stem_width, kernel_size=1, strides=1, padding="same", kernel_initializer="he_normal",
                use_bias=False, data_format="channels_last")(input_tensor)


    # Add
    x = Add()([x, shortcut])

    return x


def rsoftmax(input_tensor, filters, radix, groups):
# vorodi be _rsoftmax radix-major hast(radix ha bala va groups payin hastan): radix, groups
# shape vorodi va khoroji az r-softmax be in shekle: (batch_size, 1, 1, filters*radix) 
# chon feature-map ke az SeNet migzare varede rsoftmaax mishe

    x = input_tensor
    batch = x.shape[0]

    if radix>1:
        # print('x0', x.shape)
        x = tf.reshape(x, [-1, groups, radix, filters//groups])   # inja group-major mikonim
        # print('x2', x.shape)
        x = tf.transpose(x, [0, 2, 1, 3])              # transpose kardim ta radix-major beshe baraye r-softmax(radix-softmax)
        # yani bad az transpose, shape mishe: (batch_size, radix, groups, filters//groups) yani group_major mishe
        # print('x2', x.shape)
        x = tf.keras.activations.softmax(x, axis=1)   # yani softmax dar rastaye radix ha mizane
        # print('x3', x.shape)
        x = tf.reshape(x, [-1, 1, 1, radix*filters])     # shape in ham mishe (batch_size, 1, 1, radix*filters)
        # print('x4', x.shape)
    else:
        x = Activation('sigmoid')(x)
    return x


def SplAtConv2d(input_tensor, filters=64, kernel_size=3, dilation=1, groups=2, radix=4, active='relu',  dropout_rate = 0):
    channel_axis = -1
    x = input_tensor
    in_channels = input_tensor.shape[-1]

    x = GroupedConv2D(filters=filters * radix, kernel_size=[kernel_size for i in range(groups * radix)],
                        use_keras=True, padding="same", kernel_initializer="he_normal", use_bias=False,
                        data_format="channels_last", dilation_rate=dilation)(x)
    
    x = BatchNormalization(axis=channel_axis, epsilon=1.001e-5)(x)
    x = Dropout(dropout_rate)(x)
    x = Activation(active)(x)

    batch, rchannel = x.shape[0], x.shape[-1]
    if radix > 1:
        splited = tf.split(x, radix, axis=-1)  # ye list mishe ke len on mishe radix
        gap = sum(splited)
    else:
        gap = x

    # print('sum',gap.shape)
    gap = GlobalAveragePooling2D(data_format="channels_last")(gap)
    gap = tf.reshape(gap, [-1, 1, 1, filters])
    # print('adaptive_avg_pool2d',gap.shape)

    reduction_factor = 4
    inter_channels = max(in_channels * radix // reduction_factor, 32)

    x = Conv2D(inter_channels, kernel_size=1)(gap)   # output_shape: (batch_size, 1, 1, inter_channels)

    x = BatchNormalization(axis=channel_axis, epsilon=1.001e-5)(x)
    x = Dropout(dropout_rate)(x)
    x = Activation(active)(x)
    x = Conv2D(filters * radix, kernel_size=1)(x)
    # output_shape: (batch_size, 1, 1, filters * radix)

    atten = rsoftmax(x, filters, radix, groups)

    if radix > 1:
        logits = tf.split(atten, radix, axis=-1)
        out = sum([a * b for a, b in zip(splited, logits)])
    else:
        out = atten * x
    return out


def make_block_basic(input_tensor, filters = 64, stride=1, radix= 4, groups= 2, active='relu', dilation=1, bottleneck_width=64, dropout_rate=0.2):
    """Conv2d_BN_Relu->Bn_Relu_Conv2d
    """
        
    x = input_tensor
    channel_axis = -1
    x = BatchNormalization(axis=channel_axis, epsilon=1.001e-5)(x)
    x = Activation(active)(x)

    # Shortcut
    short_cut = x
    inplanes = input_tensor.shape[-1]               # inplanes: num_channels
    short_cut = Conv2D(filters, kernel_size=1, strides=stride, padding="same", kernel_initializer="he_normal",
                        use_bias=False, data_format="channels_last")(short_cut)          # to in spatial_dimensions nesf mishan, tedade filters ham channel darim


    ###
    group_width = filters * groups     # filters * groups : teadade filters dar split_attention

    if radix >= 1:
        x = SplAtConv2d(x, filters=group_width, kernel_size=3, dilation=dilation,
                                groups=groups, radix=radix)
    else:
        x = Conv2D(filters, kernel_size=3, padding="same", kernel_initializer="he_normal",
                    dilation_rate=dilation, use_bias=False, data_format="channels_last")(x)



    x = BatchNormalization(axis=channel_axis, epsilon=1.001e-5)(x)
    x = Dropout(dropout_rate)(x)
    x = Activation(active)(x)
    x = Conv2D(filters, kernel_size=3, strides=stride, padding="same", kernel_initializer="he_normal",
                dilation_rate=dilation, use_bias=False, data_format="channels_last")(x)

    m2 = Add()([x, short_cut])
    # print('x.shape: ', x.shape)
    return m2


input_tensor = tf.random.uniform(shape=(2, 32, 32, 8))
filters=64
radix = 4
groups=2
kernel_size = 3
dilation = 1
short_cut = input_tensor
inplanes = input_tensor.shape[-1]  
block_expansion=4
avd=True
avd_first=False
avg_down=True
stride=2
cardinality = groups
bottleneck_width=64


def build_model(input_shape):
    inputs=Input(shape=input_shape)
    c0=inputs
    n_filters=[16,32,64,128,256]
    e1 = make_stem(c0, stem_width=n_filters[0])                         # (256, 256, 16)
    c1 = s_attention(e1)

    ## Encoder
    e2 = make_block_basic(c1, filters = n_filters[1], stride=2, groups=2, radix=4)          # e2, c2 : (batch, 128, 128, 32)
    c2 = s_attention(e2)
    e3 = make_block_basic(c2, filters = n_filters[2], stride=2, groups=2, radix=4)          # e3, c3 : (batch, 64, 64, 64)
    c3 = s_attention(e3) 
    e4 = make_block_basic(c3, filters = n_filters[3], stride=2, groups=2, radix=4)          # (batch, 32, 32, 128)

    ## Bridge
    b1 = dual_attention(e4)        # b1: (batch_size, 32, 32, filters)

    ## Decoder
    d1 = attention_block(c3, b1)      # d1.shape: (batch, 32, 32, 128)
    d1 = UpSampling2D((2, 2))(d1)     # (batch, 64, 64, 128)
    d1 = Concatenate()([d1, e3])      # (batch, 64, 64, 192)
    d1 = make_block_basic(d1, filters = n_filters[3])   # (batch, 64, 64, 128)

    d2 = attention_block(e2, d1)    # (64, 64, 128)
    # print('d2.shape: ', d2.shape)
    d2 = UpSampling2D((2, 2))(d2)   # (batch, 128, 128, 128)
    # print('d2.shape: ', d2.shape)
    d2 = Concatenate()([d2, e2])   # (batch, 128, 128, 160)
    # print('d2.shape: ', d2.shape)        
    d2 = make_block_basic(d2, filters = n_filters[2])   # (batch, 128, 128, 64)
    # print('d2.shape: ', d2.shape)


    d3 = attention_block(e1, d2)   # (batch, 128, 128, 64)
    # print('d3.shape: ', d3.shape)
    d3 = UpSampling2D((2, 2))(d3)  # (batch, 256, 256, 64)
    # print('d3.shape: ', d3.shape)
    d3 = Concatenate()([d3, e1])   # (batch, 256, 256, 80)
    # print('d3.shape: ', d3.shape)
    d3 = make_block_basic(d3, filters = n_filters[1])   # (batch, 256, 256, 32)
    # print('d3.shape: ', d3.shape)

    outputs = s_attention(d3)   # (batch, 256, 256, 32)
    print('outputs.shape: ', outputs.shape)
    outputs = Conv2D(1, (1, 1), padding="same")(outputs)   # (batch, 256, 256, 1)
    outputs = Activation("sigmoid")(outputs)
    print(outputs.shape)

    ## Model
    model = Model(inputs, outputs)
    return model

if __name__ == "__main__":
    input_shape = (256, 256, 3)
    model = build_model(input_shape)
    model.summary()