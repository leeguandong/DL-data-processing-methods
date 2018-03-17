import numpy as np
from keras.applications.imagenet_utils import decode_predictions
from keras.preprocessing import image
from keras.applications import *
import glob

import os

# 忽略硬件加速的警告信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

file_path = 'Data/'
f_names = glob.glob(file_path + '*.jpg')

img = []
# 把图片读取出来放到列表中
for i in range(len(f_names)):
    images = image.load_img(f_names[i], target_size=(224, 224))
    x = image.img_to_array(images)
    x = np.expand_dims(x, axis=0)
    img.append(x)
    print('loading no.%s image' % i)

# 把图片数组联合在一起
x = np.concatenate([x for x in img])

model = ResNet50(weights='imagenet')
y = model.predict(x)
print('Predicted:', decode_predictions(y, top=3))
'''
decode_predictions 返回的是一个列表，所有batch的
# Returns
        A list of lists of top class prediction tuples
        `(class_name, class_description, score)`.
        One list of tuples per sample in batch input.
'''
# [[('n02100735', 'English_setter', 0.62966996), ('n02101556', 'clumber', 0.30187815), ('n02101388', 'Brittany_spaniel', 0.023777956)],
# [('n04589890', 'window_screen', 0.36741936), ('n03598930', 'jigsaw_puzzle', 0.16058789), ('n03000247', 'chain_mail', 0.064882509)],
# [('n02099712', 'Labrador_retriever', 0.18316899), ('n02092339', 'Weimaraner', 0.17261373), ('n02107312', 'miniature_pinscher', 0.084056087)],
# [('n02097298', 'Scotch_terrier', 0.71131712), ('n02096177', 'cairn', 0.16914253), ('n02098286', 'West_Highland_white_terrier', 0.064419992)],
# [('n02108915', 'French_bulldog', 0.95837146), ('n02110958', 'pug', 0.036947753), ('n02108089', 'boxer', 0.0012087316)],
# [('n02090721', 'Irish_wolfhound', 0.63058007), ('n02111129', 'Leonberg', 0.094695911), ('n02106382', 'Bouvier_des_Flandres', 0.086082667)],
# [('n02085936', 'Maltese_dog', 0.70757532), ('n02113624', 'toy_poodle', 0.081604019), ('n02096437', 'Dandie_Dinmont', 0.037193742)],
# [('n02088364', 'beagle', 0.41100392), ('n02092339', 'Weimaraner', 0.11080339), ('n02100236', 'German_short-haired_pointer', 0.11025359)],
# [('n02102318', 'cocker_spaniel', 0.84692949), ('n02113799', 'standard_poodle', 0.14155339), ('n02102040', 'English_springer', 0.0078855371)]]

'''
concatenate的作用是把shape为(0, 224, 224, 3)的每张图片tensor，打包成shape为(batch, 224, 224, 3)的tensor，
这样就能实现批量预测或批量训练了
'''