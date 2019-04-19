'''
颜色空间转换，如BGR - Gray,BGR - Hsv
追踪视频中特定颜色的物体
opencv函数，cv2.cvtColor(),cv2.inRange()
'''

## 颜色空间转换
import cv2

img = cv2.imread('test1.jpg')

# 转换为灰度图, 参数1是要转换的图片，参数2是转换模式
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow('img', img)
cv2.imshow('gray', img_gray)
cv2.waitKey(0)

# 显示所有的转换模式
flag = [i for i in dir(cv2) if i.startswith('COLOR_')]
print(flag)

# 经验之谈：颜色转换其实是数学运算，如灰度化最常用的是：gray=R*0.299+G*0.587+B*0.114

## 视频中追中颜色特定的物体
'''
1.捕获视频中的一帧
2.从 BGR 转换到 HSV
3.提取蓝色范围的物体
4.只显示蓝色物体
'''
import numpy as np

capture = cv2.VideoCapture(0)

# 蓝色的范围，不同光照条件下不一样，可灵活调整
lower_blue = np.array([100, 110, 110])
upper_blue = np.array([130, 255, 255])

while (True):
    # 1.捕获视频中的一帧
    ret, frame = capture.read()

    # 2.从 BGR 转换到 HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 3.InRange(): 介于 lower/upper 之间的为白色，其余黑色
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # 4.只保留原图中的蓝色部分
    res = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)
    cv2.imshow('res', res)

    if cv2.waitKey(1) == ord('q'):
        break

'''
其中，bitwise_and()函数暂时不用管，后面会讲到。那蓝色的HSV值的上下限lower和upper范围是怎么得到的呢？
其实很简单，我们先把标准蓝色的BGR值用cvtColor()转换下：

blue = np.uint8([[[255, 0, 0]]])
hsv_blue = cv2.cvtColor(blue, cv2.COLOR_BGR2HSV)
print(hsv_blue)  # [[[120 255 255]]]
'''
