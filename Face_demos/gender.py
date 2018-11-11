'''
用 keras 实现性别识别，模型数据使用的是 oarriaga/face_classification 的模型classifier

Real-time Convolutional Neural Networks for Emotion and Gener Classification
'''

'''
结合之前图片人脸检测的项目，我们使用 Opencv 先识别到人脸，然后在通过 keras 识别性别
'''

import cv2
from keras.models import load_model
import numpy as np
from opencv import chinese

img = cv2.imread("F:/Github/DL-data-processing-methods/Face_demos/img/gather.png")
face_classifier = cv2.CascadeClassifier(
    "D:\python\python 3.5\Lib\site-packages\opencv-master\data\haarcascades\haarcascade_frontalface_default.xml"
)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_classifier.detectMultiScale(
    gray, scaleFactor=1.2, minNeighbors=3, minSize=(140, 140))

gender_classifier = load_model(
    "F:/Github/DL-data-processing-methods/Face_demos/classifier/gender_models/simple_CNN.81-0.96.hdf5")
gender_labels = {0: '女', 1: '男'}
color = (255, 255, 255)

for (x, y, w, h) in faces:
    face = img[(y - 60):(y + h + 60), (x - 30):(x + w + 30)]
    face = cv2.resize(face, (48, 48))
    face = np.expand_dims(face, 0)
    face = face / 255.0
    gender_label_arg = np.argmax(gender_classifier.predict(face))
    gender = gender_labels[gender_label_arg]
    cv2.rectangle(img, (x, y), (x + h, y + w), color, 2)
    img = chinese.cv2ImgAddText(img, gender, x + h, y, color, 30)

cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
