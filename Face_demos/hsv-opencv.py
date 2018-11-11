'''
与其说是平时对象提取，不如说是视频颜色提取，因为其本质还是使用了 Opencv 的 HSV 颜色物体检测
'''
'''
HSV 分别代表，色调(H),饱和度(S),亮度(V),一种颜色空间，也称六角锥体模型。

色调: 用角度度量，取值范围是0-360，从红色开始逆时针方向计算，红色为0，绿色为120，蓝色为240.
饱和度：取值范围是0-255，值越大，颜色越饱和.
亮度：取值范围是0(黑色) - 255(白色).

实现思路：4
如上效果图所示，我们要做的就是把视频中的绿色的小猪佩奇识别出来即可，下面是识别步骤。
1.使用 PS 取小猪佩奇颜色的 HSB 值，相当于 Opencv 的HSV，
2.使用 Opencv 与运算 提取HSV的颜色部分画面.
3.使用高斯模糊优化图片
4.图片展示
'''

# HSV 转换
import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while (1):
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 在 PS 里用取色器的 HSV
    psHSV = [112, 89, 52]
    diff = 40  # 上下浮动值
    # 因为 PS 的 HSV(HSB) 取值是: 0-360,0-1,0-1, 而 Opencv 的 HSV 是：0-180,0-255,0-255，所以要对 ps 的hsv进行处理，H/2,SV*255
    lowerHSV = [(psHSV[0] - diff) / 2, (psHSV[1] - diff) * 255 / 100, (psHSV[2] - diff) * 255 / 100]
    upperHSV = [(psHSV[0] + diff) / 2, (psHSV[1] + diff) * 255 / 100, (psHSV[2] + diff) * 255 / 100]

    mask = cv2.inRange(hsv, np.array(lowerHSV), np.array(upperHSV))

    # 使用位 与运算 提取颜色部分
    res = cv2.bitwise_and(frame, frame, mask=mask)

    # 使用高斯模式优化图片
    res = cv2.GaussianBlur(res, (5, 5), 1)

    cv2.imshow('frame', frame)
    cv2.imshow('res', res)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
