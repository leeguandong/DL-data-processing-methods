# -*- coding: utf-8 -*-
'''
https://github.com/1zlab/1ZLAB_Face_Track_Robot

读取手机摄像头的视频流
手机与电脑链接同一个WIFI热点
'''
import cv2
import face_recognition
import os

path = 'F:/Github/DL-data-processing-methods/Demos/img/face_recognition'  # 模拟数据图片目录
cap = cv2.VideoCapture(0)
total_image_name = []
total_face_encoding = []

for fn in os.listdir(path):  # fn表示的是文件名 q
    print(path + '/' + fn)
    total_face_encoding.append(face_recognition.face_encodings(face_recognition.load_image_file(path + '/' + fn))[0])
    fn = fn[:(len(fn) - 4)]  # 截取图片名(这里应该是 images 文件中的图片名命名为人物名)
    total_image_name.append(fn)  # 图片命名字列表

ip_camera_url = 'http://admin:admin@192.168.137.134:8081/'
# ip_camera_url = 'http://admin:admin@192.168.137.66:8081/'

# 创建一个窗口
cv2.namedWindow('ip_camera', flags=cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)

cap = cv2.VideoCapture(ip_camera_url)

# 设置缓存区的大小, cap.set 设定VideoCapture的各种属性，这里指定缓冲区的尺寸为1
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print('请检查IP地址还有端口号，或者查看IP摄像头是否开启，另外记得使用sudo权限运行脚本')

while cap.isOpened():
    ret, frame = cap.read()

    # 发现在视频帧所有的脸和 face_encodings
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    # 在这个视频帧中循环遍历每个人脸
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # 看看面部是否与已知的人脸相匹配
        for i, v in enumerate(total_face_encoding):
            match = face_recognition.compare_faces([v], face_encoding, tolerance=0.5)
            name = 'Unknown'
            if match[0]:
                name = total_image_name[i]
                break
        # 画出一个框，框柱脸
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        # 画出一个带有名字的标签，放在框下
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # 显示结果图像
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyWindow('ip_camera')
cap.release()
