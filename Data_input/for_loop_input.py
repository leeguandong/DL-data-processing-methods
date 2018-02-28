'''
与多张图片输入最大的不用，在于输入图片并没有concatenate形成一个batch，从而批量进行训练或者预测。
这里是一张一张输入到一个list中，然后预测时，从list1中一个一个的取出来，送到模型中去预测，模型
本身是不定输入尺寸的，一张一张输入就避免了在同一个batch中模型考虑用哪一个尺寸作为输出的问题。
'''
from keras.applications.imagenet_utils import decode_predictions
from keras.preprocessing import image
from keras.applications import *
import numpy as np
import glob

import os

# 忽略硬件加速的警告信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

file_path = 'Data/'

def read_image(file_path):
    img = []
    f_names = glob.glob(file_path + '*.jpg')
    for i in range(len(f_names)):
        images = image.load_img(f_names[i], target_size=(224, 224))
        x = image.array_to_img(images)
        x = np.expand_dims(x, axis=0)
        img.append(x)
        print('loading no.%s image' % i)
    return img

def predict_image(model, img):
    pred = []
    for i in range(len(img)):
        y = model.predict(img[i], batch_size=1)
        print('Predicted:', decode_predictions(y, top=3))
        pred.append(y)
    return pred

model = ResNet50()
img = read_image(file_path)
y = predict_image(model, img)
