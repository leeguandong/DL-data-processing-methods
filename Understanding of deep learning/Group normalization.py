# coding=utf-8
'''
@author leeguandon
paper: Group Normalization
'''

from keras.engine import Layer, InputSpec
from keras import initializers
from keras import regularizers
from keras import constraints
from keras import backend as K

from keras.utils.generic_utils import get_custom_objects
import tensorflow as tf


class GroupNormalization(Layer):
    '''
    Group normalization divides the channels into groups and computes within each group the mean and variance for normalization
    GN's comptation is independent of batch sizes,and its accuracy is stable in a wid range of batch sizes
    '''

    def __init__(self,
                 group=32,
                 axis=-1,
                 momentum=0.99,
                 epsilon=1e-5,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 moving_mean_initializer='zeros',
                 ):
        pass
