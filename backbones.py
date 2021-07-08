from keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, ReLU, add, GlobalAveragePooling2D, \
                         Reshape, Dense, multiply, UpSampling2D
from keras.models import Model
import keras.backend as K


def resnet(input_shape=(512,512,3), depth=50, use_fpn=False):

    n_blocks = {50: [3,4,6,3], 101: [3,4,23,3]}
    n_filters = [256, 512, 1024, 2048]

    inpt = Input(input_shape)

    # stem: conv+bn+relu+pool
    x = Conv_BN(inpt, 64, 7, strides=2, activation='leaky')
    x = MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    # blocks
    num_blocks = n_blocks[depth]
    features = []      # P2-P5
    for i in range(len(num_blocks)):
        for j in range(num_blocks[i]):
            strides = 2 if i!=0 and j==0 else 1
            x = res_block(x, n_filters[i], strides)
        features.append(x)

    # model
    if use_fpn:
        fpn_features = fpn(features[1:])       # Retina-style fpn
        model = Model(inpt, fpn_features)
    else:
        model = Model(inpt, x)
    model.name = 'resnet'
    model.load_weights("weights/r50.h5", by_name=True, skip_mismatch=True)     # load_pretrained

    return model


def hourglass(input_shape=(512,512,3), n_stacks=2, n_classes=16):

    n_channles=[256, 384, 384, 384, 512]

    inpt = Input(input_shape)

    # stem: 7x7 s2 conv + s2 residual
    x = Conv_BN(inpt, 128, 3, strides=2, activation='relu')
    x = residual(x, 256, strides=2)

    # hourglass modules
    outputs = []
    for i in range(n_stacks):
        x, x_intermediate = hourglass_module(x, n_classes, n_channles)
        if i!= n_stacks-1:
            outputs.append(x_intermediate)

    # model
    model = Model(inpt, [x]+outputs)
    model.name = 'hourglass'

    return model


def fpn(features, filters=256):
    # lateral connections(1x1 conv)
    features = [Conv2D(filters, 1, strides=1,padding='same',activation=None)(i) for i in features]
    C3, C4, C5 = features
    # top-down connections(upSampling)
    P5 = C5
    P5_up = UpSampling2D(size=2, interpolation='nearest')(P5)
    P4 = add([C4, P5_up])
    P4_up = UpSampling2D(size=2, interpolation='nearest')(P4)
    P3 = add([C3, P4_up])
    # p6p7
    P6 = Conv2D(filters, 3, strides=2,padding='same',activation=None)(P5)
    P7 = Conv2D(filters, 3, strides=2,padding='same',activation=None)(ReLU()(P6))
    return [P3,P4,P5,P6,P7]


def res_block(x, n_filters, strides):
    inpt = x
    # residual
    x = Conv_BN(x, n_filters//4, 1, strides=strides, activation='relu')
    x = Conv_BN(x, n_filters//4, 3, strides=1, activation='relu')
    x = Conv_BN(x, n_filters, 1, strides=1, activation=None)
    # shortcut
    if strides!=1 or inpt._keras_shape[-1]!=n_filters:
        inpt = Conv_BN(inpt, n_filters, 1, strides=strides, activation=None)
    x = add([inpt, x])
    x = ReLU()(x)
    return x


def Conv_BN(x, n_filters, kernel_size, strides, activation=None):
    x = Conv2D(n_filters, kernel_size, strides=strides, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    if activation:
        x = ReLU()(x)
    return x


def hourglass_module(inpt, n_classes, n_channles):
    x = inpt
    features = [x]     # from x4 to x64
    n_levels = len(n_channles)
    # encoder: s1 residual + s2 residual
    for i in range(n_levels):
        if i!=0:
            x = residual(x, n_channles[i], strides=1)
            features.append(x)
        if i!=n_levels-1:
            x = residual(x, n_channles[i+1], strides=2)

    # mid connection: 4 residuals
    for i in range(4):
        x = residual(x, n_channles[-1], strides=1)

    # decoder:
    for i in range(n_levels-2, -1, -1):
        # skip: 2 residuals
        skip = features[i]
        skip = residual(skip, n_channles[i], strides=1)
        skip = residual(skip, n_channles[i], strides=1)
        # features
        x = residual(x, n_channles[i])
        x = residual(x, n_channles[i])
        x = UpSampling2D()(x)
        # add
        x = add([x, skip])

    # head branches
    x_intermediate = Conv2D(n_classes, 1, strides=1, padding='same', activation='sigmoid')(x)
    input_branch = Conv2D(n_channles[0], 1, strides=1, padding='same', use_bias=False)(inpt)
    output_branch = Conv2D(n_channles[0], 1, strides=1, padding='same', use_bias=False)(x)
    x = add([input_branch, output_branch])
    x = ReLU()(x)

    return x, x_intermediate


def residual(inpt, n_filters, strides=1):
    # bottleneck: 1x1, 3x3, 1x1
    x = inpt
    x = Conv_BN(x, n_filters//2, 1, strides=strides, activation='relu')
    x = Conv_BN(x, n_filters//2, 3, strides=1, activation='relu')
    x = Conv_BN(x, n_filters, 1, strides=1, activation=None)
    # skip: 1x1 conv
    if K.int_shape(inpt)[-1] == n_filters and strides==1:
        # identity
        skip = inpt
    else:
        # 1x1 conv
        skip = Conv_BN(inpt, n_filters, 1, strides=strides, activation=None)
    # add
    x = add([x, skip])
    x = ReLU()(x)
    return x



