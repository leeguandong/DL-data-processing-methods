'''
实现思路：调用电脑的摄像头，把摄像的信息逐帧分解成图片，基于图片检测标识出人脸的位置，把处理的图片逐帧绘制给用户，用户看到的
效果就是视频的人脸检测。
'''

import cv2


# 图片识别方法封装
def discern(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cap = cv2.CascadeClassifier(
        'D:\python\python 3.5\Lib\site-packages\opencv-master\data\haarcascades\haarcascade_frontalface_default.xml')
    faceRects = cap.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(45, 45))
    if len(faceRects):
        for faceRect in faceRects:
            x, y, w, h = faceRect
            cv2.rectangle(img, (x, y), (x + h, y + w), (0, 255, 0), 2)  # 框出人脸
    # cv2.resizeWindow("Image", 640, 640)
    cv2.imshow('Image', img)



# 获取摄像头 0 表示第一个摄像头
cap = cv2.VideoCapture(0)
while (1):  # 逐帧显示
    ret, img = cap.read()
    discern(img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()  # 摄像头
cv2.destroyAllWindows()  # 释放窗口资源
