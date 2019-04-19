import os
import sys
import skimage.io
import matplotlib.pyplot as plt

# Root directory of the priject
ROOT_DIR = os.path.abspath('F:/Github/DL-data-processing-methods/Understanding of deep learning/MaskRCNN/')

# Import Mask RCNN
sys.path.append(ROOT_DIR)
from MaskRCNN import utils
import MaskRCNN.model as modellib
from MaskRCNN import visualize

# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, 'samples/coco/'))
import MaskRCNN.coco

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, 'logs')

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, 'mask_rcnn_coco.h5')
# Download COCO trained weights from Release if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, 'Understanding of deep learning/MaskRCNN/images')
