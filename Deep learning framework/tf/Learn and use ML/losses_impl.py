# Implementation of Loss operations for use in neural networks
from tensorflow.python.util.tf_export import tf_export
# Utilities for exporting TensorFlow symbols to the API

@tf_export('losses.Reduction')
class Reduction(object):
      # Types of loss reduction
      pass

def _safe_div(numerator,denominator,name='value'):
    # 计算安全分区，如果分母为零，则返回0
    pass

def _safe_mean(losses,num_present):
    # computes a safe mean of the losses
    pass

def _num_present(losses,weights,per_batch=False):


