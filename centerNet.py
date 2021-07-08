from keras.layers import Input, Conv2D, BatchNormalization, ReLU, UpSampling2D, add, Lambda, \
                         GlobalAveragePooling2D, Dense, multiply, Reshape
from keras.models import Model
import keras.backend as K
import tensorflow as tf
from keras.engine import Layer
from backbones import resnet, hourglass
from GroupNormalization import GroupNormalization
from loss import centerNet_loss


# backs = {'r50-fpn': resnet(input_shape=(512,512,3),depth=50,fpn=True),
#          'hourglass': hourglass(input_shape=(512,512,3), n_stacks=2, n_classes=16),
#          # 'dla34': , rx101,
#          }


def centerNet(input_shape=(512,512,3), n_classes=16, levels=5):
    inpt = Input(input_shape)

    # backbone
    backbone = resnet(input_shape=(512,512,3),depth=50,use_fpn=True)
    features = backbone(inpt)[:levels]    # P3-P7

    # task-individual convs
    cls_branch_features = cls_branch(features)
    box_branch_features = box_branch(features)

    # heads
    cls_heads = cls_head(cls_branch_features, n_classes)   # [b,h,w,C] stage2-clsprob
    box_heads = box_head(box_branch_features)              # [b,h,w,4] stage2-boxreg
    cls_agn_heads = cls_agn_head(box_branch_features)      # [b,h,w,1] stage1-prob / objectness / centerness 

    # loss
    strides = [8,16,32,64,128]
    h, w = input_shape[:2]
    gts = [Input((h//s,w//s)+(n_classes+1+4,)) for s in strides[:levels]]     # [b,h,w,C+1+4]
    loss = Lambda(centerNet_loss, arguments={'levels': levels, 'n_classes': n_classes}, name='centerNet_loss')(
                  [*cls_heads, *box_heads, *cls_agn_heads, *gts])

    # model
    model = Model([inpt,*gts], loss)

    return model


def cls_branch(feats):
    # 4 convs-GN-relu
    def _cls_branch(filters=256):
        inpt = Input((None, None, 256))
        x = Conv2D(filters, 3, strides=1, padding='same', use_bias=True, activation=None)(inpt)
        x = GroupNormalization(groups=32)(x)
        x = ReLU()(x)
        return Model(inpt, x, name='cls_branch')

    func = _cls_branch(filters=256)
    cls_features = []
    for feat in feats:
        cls_features.append(func(feat))
    return cls_features


def box_branch(feats):
    # 4 convs-GN-relu
    def _box_branch(filters=256):
        inpt = Input((None, None, 256))
        x = Conv2D(filters, 3, strides=1, padding='same', use_bias=True, activation=None)(inpt)
        x = GroupNormalization(groups=32)(x)
        x = ReLU()(x)
        return Model(inpt, x, name='box_branch')

    func = _box_branch(filters=256)
    box_features = []
    for feat in feats:
        box_features.append(func(feat))
    return box_features


class Scale(Layer):
    # FCOS learnable scalar
    def __init__(self, init_value=1., **kwargs):
        super(Scale, self).__init__(**kwargs)
        self.scale = K.variable(init_value, dtype='float32')

    def call(self, x):
        return self.scale * x

    def compute_output_shape(self, input_shape):
        return input_shape


def cls_head(feats, n_classes):
    # 3x3 conv  ->  logits
    def _cls_head():
        inpt = Input((None, None, 256))
        x = Conv2D(n_classes, 3, strides=1, padding='same', use_bias=True, activation=None)(inpt)
        return Model(inpt, x, name='cls_head')

    func = _cls_head()
    cls_heads = []
    for feat in feats:
        cls_heads.append(func(feat))
    return cls_heads


def box_head(feats):
    def _box_head():
        inpt = Input((None, None, 256))
        x = Conv2D(4, 3, strides=1, padding='same', use_bias=True, activation=None)(inpt)
        return Model(inpt, x, name='box_head')

    func = _box_head()
    box_heads = []
    for feat in feats:
        box_heads.append(ReLU()(Scale()(func(feat))))
    return box_heads


def cls_agn_head(feats):
    def _cls_agn_head():
        inpt = Input((None, None, 256))
        x = Conv2D(1, 1, strides=1, padding='same', use_bias=True, activation=None)(inpt)
        return Model(inpt, x, name='cls_agn_head')

    func = _cls_agn_head()
    cls_agn_heads = []
    for feat in feats:
        cls_agn_heads.append(func(feat))
    return cls_agn_heads


if __name__ == '__main__':

    model = centerNet()
    model.summary()











