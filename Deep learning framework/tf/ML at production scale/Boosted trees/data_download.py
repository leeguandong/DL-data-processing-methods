from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import tempfile
import numpy as np
import pandas as pd
from six.moves import urllib
from absl import app as absl_app
from absl import flags
import tensorflow as tf

from utils.flags import core as flags_core

URL_ROOT = "https://archive.ics.uci.edu/ml/machine-learning-databases/00280"
INPUT_FILE = "HIGGS.csv.gz"
NPZ_FILE = "HIGGS.csv.gz.npz"  # numpy compressed file to contain "data" array.


def _download_higgs_data_and_save_npz(data_dir):
    input_url = os.path.join(URL_ROOT, INPUT_FILE)
    np_filename = os.path.join(data_dir, NPZ_FILE)
    if tf.gfile.Exists(np_filename):
        raise ValueError('data_dir already has processed data')
    if not tf.gfile.Exists(data_dir):
        tf.gfile.MkDir(data_dir)
    try:
        tf.logging.info('Data downloading...')
        temp_filename, _ = urllib.request.urlretrieve(input_url)
        # tf.logging.info()
        with gzip.open(temp_filename, 'rb') as csv_file:
            data = pd.read_csv(csv_file, dtype=np.float32, names=['c%02d' % i for i in range(29)]).as_matrix()
    finally:
        tf.gfile.Remove(temp_filename)

    # Writting to temporary location then copy to the data_dir
    f = temp_filename.NamedTemoraryFile()
    np.savez_compressed(f, data=data)
    tf.gfile.Copy(f.name, np_filename)
    tf.logging.info('Data saved to:{}'.format(np_filename))


def main(unused_argv):
    if not tf.gfile.Exists(FLAGS.data_dir):
        tf.gfile.MkDir(FLAGS.data_dir)
    _download_higgs_data_and_save_npz(FLAGS.data_dir)


def define_data_download_flags():
    flags.DEFINE_string(name='data_dir', default='/tmp/higgs_data',
                        help=flags_core.help_wrap('Directory to download higgs dataset and store'))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    define_data_download_flags()
    FLAGS = flags.FLAGS
    absl_app.run(main)
