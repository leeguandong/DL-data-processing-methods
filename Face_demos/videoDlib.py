'''
1. Dlib 模型识别的准确率和效果要好于 Opencv
2. Dlib 识别的性能要比 Opencv 差，使用视频测试的时候 Dlib 有明显的卡顿，但是 Opencv 就好很多
'''

import cv2
import dlib

detector = dlib.get_frontal_face_detector()  # 使用默认的人脸识别模型


def discern(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dets = detector(gray, 1)
    for face in dets:
        left = face.left()
        top = face.top()
        right = face.right()
        bottom = face.bottom()
        cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.imshow('Image', img)


cap = cv2.VideoCapture(0)
while (1):
    ret, img = cap.read()
    discern(img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
