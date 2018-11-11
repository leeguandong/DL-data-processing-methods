'''
表情识别：表情识别支持 7 种表情类型，生气、厌恶、恐惧、开心、难过、惊喜、平静等
Real-time Convolutional Neural Networks for Emotion and Gener Classification
'''

import cv2
from keras.models import load_model
import numpy as np
from opencv import chinese
import datetime

startTime = datetime.datetime.now()
emotion_classifier = load_model(
    'F:/Github/DL-data-processing-methods/Face_demos/classifier/emotion_models/simple_CNN.530-0.65.hdf5')
endTime = datetime.datetime.now()
print(endTime - startTime)

emotion_labels = {
    0: '生气',
    1: '厌恶',
    2: '恐惧',
    3: '开心',
    4: '难过',
    5: '惊喜',
    6: '平静'
}

img = cv2.imread('F:/Github/DL-data-processing-methods/Face_demos/img/face_recognition/wa.jpg')
face_classifier = cv2.CascadeClassifier(
    'D:\python\python 3.5\Lib\site-packages\opencv-master\data\haarcascades\haarcascade_frontalface_default.xml')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_classifier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(40, 40))
color = (255, 0, 0)

for (x, y, w, h) in faces:
    gray_face = gray[(y):(y + h), (x):(x + w)]
    gray_face = cv2.resize(gray_face, (48, 48))
    gray_face = gray_face / 255.0
    gray_face = np.expand_dims(gray_face, 0)
    gray_face = np.expand_dims(gray_face, -1)
    emotion_label_arg = np.argmax(emotion_classifier.predict(gray_face))
    emotion = emotion_labels[emotion_label_arg]
    cv2.rectangle(img, (x + 10, y + 10), (x + h - 10, y + w - 10), (0, 255, 0), 2)
    img = chinese.cv2ImgAddText(img, emotion, x + h * 0.4, y - w * 0.06, color, 20)

cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
