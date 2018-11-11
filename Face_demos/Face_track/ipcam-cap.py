# -*- coding: utf-8 -*-
'''
https://github.com/1zlab/1ZLAB_Face_Track_Robot

读取手机摄像头的视频流
手机与电脑链接同一个WIFI热点
'''
import cv2
import dlib

ip_camera_url = 'http://admin:admin@192.168.137.134:8081/'
# ip_camera_url = 'http://admin:admin@192.168.137.66:8081/'

# 创建一个窗口
cv2.namedWindow('ip_camera', flags=cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)

cap = cv2.VideoCapture(ip_camera_url)

# 设置缓存区的大小, cap.set 设定VideoCapture的各种属性，这里指定缓冲区的尺寸为1
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print('请检查IP地址还有端口号，或者查看IP摄像头是否开启，另外记得使用sudo权限运行脚本')

detector = dlib.get_frontal_face_detector()  # 使用默认的人脸识别模型


def discerndlip(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dets = detector(gray, 1)
    for face in dets:
        left = face.left()
        top = face.top()
        right = face.right()
        bottom = face.bottom()
        cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.imshow('Image', img)


# 图片识别方法封装
def discernopencv(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cap = cv2.CascadeClassifier(
        'D:\python\python 3.5\Lib\site-packages\opencv-master\data\haarcascades\haarcascade_frontalface_default.xml')
    faceRects = cap.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(50, 50))
    if len(faceRects):
        for faceRect in faceRects:
            x, y, w, h = faceRect
            cv2.rectangle(img, (x, y), (x + h, y + w), (0, 255, 0), 2)  # 框出人脸
    cv2.imshow('Image', img)


while cap.isOpened():
    ret, frame = cap.read()
    discernopencv(frame)
    # discerndlip(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # cv2.imshow('ip_camera', frame)

    if cv2.waitKey(1) == ord('q'):
        # 退出程序
        break

cv2.destroyWindow('ip_camera')
cap.release()
