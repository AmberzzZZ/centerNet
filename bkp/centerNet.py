from keras.layers import Input, Conv2D, BatchNormalization, ReLU, UpSampling2D, add, Lambda, \
                         GlobalAveragePooling2D, Dense, multiply, Reshape
from keras.models import Model
import keras.backend as K
from loss import det_loss, inter_loss


def centerNet(input_shape=(512,512,3), n_classes=80, n_stacks=2, n_channles=[256, 384, 384, 384, 512]):
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

    # kp prediction: separated 3x3 conv & 1x1 conv
    head = x
    # heatmap branch
    x = Conv_BN(head, n_channles[-1], 3, 1, activation='relu')
    heatmaps = Conv2D(n_classes, 1, strides=1, padding='same', activation='sigmoid')(x)
    # offset branch
    x = Conv_BN(head, n_channles[-1], 3, 1, activation='relu')
    offsets = Conv2D(2, 1, strides=1, padding='same', activation='sigmoid')(x)
    # size branch
    x = Conv_BN(head, n_channles[-1], 3, 1, activation='relu')
    sizes = Conv2D(2, 1, strides=1, padding='same', activation=None)(x)

    # loss
    gt = Input((input_shape[0]//4, input_shape[1]//4, n_classes+2+2))
    loss = Lambda(det_loss, arguments={'n_classes':n_classes})([heatmaps, offsets, sizes, gt])
    inter_sv = Lambda(inter_loss, arguments={'n_classes':n_classes})([*outputs, gt])

    # model
    model = Model([inpt, gt], [loss, inter_sv])

    return model


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


def residual(inpt, n_filters, strides=1, se_ratio=0):
    # bottleneck: 1x1, 3x3, 1x1
    x = inpt
    x = Conv_BN(x, n_filters//2, 1, strides=strides, activation='relu')
    x = Conv_BN(x, n_filters//2, 3, strides=1, activation='relu')
    x = Conv_BN(x, n_filters, 1, strides=1, activation=None)
    # se-block
    if se_ratio and strides==1:
        x = SE_block(x, se_ratio)
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


def Conv_BN(x, n_filters, kernel_size, strides, activation=None):
    x = Conv2D(n_filters, kernel_size, strides=strides, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    if activation:
        x = ReLU()(x)
    return x


def SE_block(inpt, ratio=16):     # spatial squeeze and channel excitation
    x = inpt
    n_filters = x._keras_shape[-1]
    x = GlobalAveragePooling2D()(x)
    x = Reshape((1,1,n_filters))(x)
    x = Dense(n_filters//ratio, activation='relu', use_bias=False)(x)
    x = Dense(n_filters, activation='sigmoid', use_bias=False)(x)
    x = multiply([inpt, x])
    return x


def sSE_block(inpt):        # channel squeeze and spatial excitation
    x = Conv2D(1, kernel_size=1, activation='sigmoid')(inpt)
    x = multiply([inpt, x])
    return x


def scSE_block(inpt, ratio=16):
    x1 = SE_block(inpt, ratio)
    x2 = sSE_block(inpt)
    x = add([x1,x2])
    return x


if __name__ == '__main__':

    model = centerNet()
    model.summary()
    # model.save("hourglass_corner.h5")









