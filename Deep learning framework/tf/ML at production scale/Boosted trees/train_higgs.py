'''
tf.estimator.BoostedTreesClassifier
树模型分类器
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
from absl import app as absl_app
from absl import flags
import tensorflow as tf

from utils.flags import core as flags_core
from utils.flags._conventions import help_wrap
from utils.logs import logger

NPZ_FILE = 'HIGGS.csv.gz.npz'

def read_higgs_data(data_dir,train_start,train_count,eval_start,eval_count):
    # Reads higgs data from csv and return train and eval data
    npz_filename = os.path.join(data_dir,NPZ_FILE)
    try:
        with tf.gfile.Open(npz_filename,'rb') as npz_file:
            with np.load(npz_file) as npz:

