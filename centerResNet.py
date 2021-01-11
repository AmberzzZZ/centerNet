# resnet 50 & 101
from keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, ReLU, add, UpSampling2D, \
                         Reshape, Dense, multiply, GlobalAveragePooling2D, Lambda
from keras.models import Model
import keras.backend as K
from loss import det_loss, inter_loss


n_blocks = [3,4,4,6,3,3]   # deeper
n_filters=[256, 384, 384, 384, 512, 1024]


def resnet(input_shape=(512,512,3), depth=50, n_stacks=2, n_classes=15):
    inpt = Input(input_shape)

    # stem: conv+bn+relu+pool
    x = Conv_BN(inpt, 64, 7, strides=2, activation='leaky')
    x = MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    # hourglass modules
    inter_outputs = []
    for i in range(n_stacks):
        x, x_intermediate = hourglass_module(x, n_classes, n_filters)
        if i!= n_stacks-1:
            inter_outputs.append(x_intermediate)

    # kp prediction: separated 3x3 conv & 1x1 conv
    head = x
    # heatmap branch
    x = Conv_BN(head, n_filters[-1], 3, 1, activation='relu')
    heatmaps = Conv2D(n_classes, 1, strides=1, padding='same', activation='sigmoid')(x)
    # offset branch
    x = Conv_BN(head, n_filters[-1], 3, 1, activation='relu')
    offsets = Conv2D(2, 1, strides=1, padding='same', activation='sigmoid')(x)
    # size branch
    x = Conv_BN(head, n_filters[-1], 3, 1, activation='relu')
    sizes = Conv2D(2, 1, strides=1, padding='same', activation=None)(x)

    # loss
    gt = Input((input_shape[0]//4, input_shape[1]//4, n_classes+2+2))
    loss = Lambda(det_loss, arguments={'n_classes':n_classes})([heatmaps, offsets, sizes, gt])
    inter_sv = Lambda(inter_loss, arguments={'n_classes':n_classes})([*inter_outputs, gt])

    # model
    model = Model([inpt, gt], [loss, inter_sv])

    return model


def hourglass_module(inpt, n_classes, n_filters):
    # resnet downsamp
    x = inpt
    features = []     # from x4 to x128
    for i in range(len(n_blocks)):
        for j in range(n_blocks[i]):
            strides = 2 if i!=0 and j==0 else 1
            x = res_block(x, n_filters[i], strides)
        features.append(x)

    # mid connection: 4 residuals
    for i in range(2):
        x = res_block(x, n_filters[-1], strides=1)

    # decoder:
    for i in range(len(n_blocks)-2, -1, -1):
        # skip: 2 residuals
        skip = features[i]
        # features
        x = UpSampling2D()(x)
        x = res_block(x, n_filters[i])
        x = res_block(x, n_filters[i])
        # add
        x = add([x, skip])

    # head branches
    x_intermediate = Conv2D(n_classes, 1, strides=1, padding='same', activation='sigmoid')(x)
    input_branch = Conv2D(n_filters[0], 1, strides=1, padding='same', use_bias=False)(inpt)
    output_branch = Conv2D(n_filters[0], 1, strides=1, padding='same', use_bias=False)(x)
    x = add([input_branch, output_branch])
    x = ReLU()(x)

    return x, x_intermediate


def res_block(x, n_filters, strides=1, se_ratio=16):
    inpt = x
    # residual
    x = Conv_BN(x, n_filters//4, 1, strides=strides, activation='relu')
    x = Conv_BN(x, n_filters//4, 3, strides=1, activation='relu')
    x = Conv_BN(x, n_filters, 1, strides=1, activation=None)
    if se_ratio:
        x = SE_block(x, se_ratio)
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


def SE_block(inpt, ratio=16):     # spatial squeeze and channel excitation
    x = inpt
    n_filters = x._keras_shape[-1]
    x = GlobalAveragePooling2D()(x)
    x = Reshape((1,1,n_filters))(x)
    x = Dense(n_filters//ratio, activation='relu', use_bias=False)(x)
    x = Dense(n_filters, activation='sigmoid', use_bias=False)(x)
    x = multiply([inpt, x])
    return x


def eff_SE_block(x, ratio=16):
    inpt = x
    n_filters = x._keras_shape[-1]
    # squeeze
    x = GlobalAveragePooling2D()(x)
    x = Reshape((1,1,n_filters))(x)
    # reduce
    x = Conv2D(n_filters//ratio, 1, strides=1, padding='same', activation=swish, use_bias=False)(x)
    # excite
    x = Conv2D(n_filters, 1, strides=1, padding='same', activation='sigmoid', use_bias=False)(x)
    # reweight
    x = multiply([inpt, x])
    return x


def swish(x):
    return x * K.sigmoid(x)


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

    model = resnet(input_shape=(512,512,3), depth=50)
    model.summary()
    # model.load_weights("resnet50_weights_tf_dim_ordering_tf_kernels.h5")

