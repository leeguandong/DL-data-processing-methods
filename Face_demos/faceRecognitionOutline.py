'''
使用 face_rocognition 绘制脸部特征，脸部轮廓
'''

import face_recognition
from PIL import Image, ImageDraw

# 将图片文件加载到 numpy 数组中
image = face_recognition.load_image_file('F:/Github/DL-data-processing-methods/Demos/img/meitu_00001.jpg')

# 查找图像中所有的面部特征
face_landmarks_list = face_recognition.face_landmarks(image)

for face_landmarks in face_landmarks_list:
    facial_features = ['chin', 'left_eyebrow', 'right_eyebrow', 'nose_bridge', 'nose_tip', 'left_eye', 'right_eye',
                       'top_lip', 'bottom_lip']
    pil_image = Image.fromarray(image)
    d = ImageDraw.Draw(pil_image)
    for facial_feature in facial_features:
        d.line(face_landmarks[facial_feature], fill=(255, 255, 255), width=3)
    pil_image.show()
