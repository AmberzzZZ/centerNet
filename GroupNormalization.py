from keras.engine import Layer, InputSpec
from keras import initializers
from keras import regularizers
from keras import constraints
import keras.backend as K
from keras.utils.generic_utils import get_custom_objects
import tensorflow as tf


class GroupNormalization(Layer):

    def __init__(self,
                 groups=32,
                 axis=-1,
                 momentum=0.99,
                 epsilon=1e-5,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        super(GroupNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.groups = groups
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

    def build(self, input_shape):
        dim = input_shape[self.axis]

        if dim is None:
            raise ValueError('Axis ' + str(self.axis) + ' of '
                             'input tensor should have a defined dimension '
                             'but the layer received an input with shape ' +
                             str(input_shape) + '.')

        if dim < self.groups:
            raise ValueError('Number of groups (' + str(self.groups) + ') cannot be '
                             'more than the number of channels (' +
                             str(dim) + ').')

        if dim % self.groups != 0:
            raise ValueError('Number of groups (' + str(self.groups) + ') must be a '
                             'multiple of the number of channels (' +
                             str(dim) + ').')

        self.input_spec = InputSpec(ndim=len(input_shape),
                                    axes={self.axis: dim})

        if len(input_shape)==5:
            # 3-dims
            shape_ = (1,dim,1,1,1)
        else:
            # 2-dims
            shape_ = (1,dim,1,1)

        if self.scale:
            self.gamma = self.add_weight(shape=shape_,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)

        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(shape=shape_,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)

        else:
            self.beta = None

        self.built = True

    def call(self, inputs, training=None, **kwargs):

        G = self.groups

        # transpose:[b,***,c] -> [b,c,***]
        if K.shape(inputs).shape[0] == 5:
            # 3-dim
            if self.axis in {-1,4}:
                inputs = K.permute_dimensions(inputs,(0,4,1,2,3))
            N, C, D, H, W = K.int_shape(inputs)
            inputs = K.reshape(inputs,(-1, G, C//G, D, H, W))
            # compute group-channel mean & variance
            gn_mean = K.mean(inputs,axis=[2,3,4,5],keepdims=True)
            gn_variance = K.var(inputs,axis=[2,3,4,5],keepdims=True)

            outputs = (inputs - gn_mean) / (K.sqrt(gn_variance + self.epsilon))
            outputs = K.reshape(outputs,[-1, C, D, H, W]) * self.gamma + self.beta

            # transpose back:
            if self.axis in {-1,4}:
                outputs = K.permute_dimensions(outputs,(0,2,3,4,1))

        else:
            # 2-dim
            x = K.permute_dimensions(inputs,(0,3,1,2))
            shape_back = tf.shape(x)
            N, C, H, W = K.int_shape(x)
            b = tf.shape(x)[0]
            x = K.reshape(x,(b,G,-1))
            # compute group-channel mean & variance
            gn_mean = K.mean(x,axis=[2],keepdims=True)
            gn_variance = K.var(x,axis=[2],keepdims=True)

            outputs = (x - gn_mean) / (K.sqrt(gn_variance + self.epsilon))
            outputs = K.reshape(outputs,shape_back) * self.gamma + self.beta

            # transpose back:
            outputs = K.permute_dimensions(outputs,(0,2,3,1))

        return outputs

    def get_config(self):
        config = {
            'groups': self.groups,
            'axis': self.axis,
            'momentum': self.momentum,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint)
        }
        base_config = super(GroupNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


get_custom_objects().update({'GroupNormalization': GroupNormalization})


if __name__ == '__main__':

    from keras.layers import Input
    x = Input((96,128,128,32))
    x = Input((128,128,32))
    x = Input((None, None, 32))
    y = GroupNormalization(groups=16)(x)
    print(y)




