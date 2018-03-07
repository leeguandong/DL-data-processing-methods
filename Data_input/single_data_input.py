import numpy as np
from keras.applications.imagenet_utils import decode_predictions
from keras.preprocessing import image
from keras.applications import *

import os

# 忽略硬件加速的警告信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

file_path = 'images/glasses.png'

img = image.load_img(file_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

model = ResNet50(weights='imagenet')
y = model.predict(x)

# print(np.argmax(y))
print('Predicted:', decode_predictions(y, top=3)[0])
# (class_name, class_description, score)
# Predicted: [('n02097298', 'Scotch_terrier', 0.71131718), ('n02096177', 'cairn', 0.16914254), ('n02098286', 'West_Highland_white_terrier', 0.06442)]

'''
解释下步骤，load_img是keras调用了pillow的Image函数，对指定file_path的图片进行提取，然后使用img_to_array把图片转换成
为numpy数组，shape为(224, 224, 3)，而expand_dims的作用是把shape(224, 224, 3)转换成(0, 224, 224, 3)，为什么要这个
expand_dims？因为模型本身要求输入尺寸是(None, 224, 224, 3)，这个None表示batch，意思是你要输入多少张图片模型是不知道的，
所以就用None来表示，而当你输入图片的时候，shape必须跟模型输入的shape保持一致才能输入。
'''
