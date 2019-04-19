'''
绘制各种几何形状，添加文字
opencv函数： cv2.line(),cv2.circle().cv2.rectangle(),cv2.ellipse(),cv2.putText()
'''
import cv2
import numpy as np

## 画线
# 创建一副黑色的图片
img = np.zeros((512, 512, 3), np.uint8)

# 画一条线宽为5的蓝色直线，参数2：起点，参数3：终点
cv2.line(img, (0, 0), (512, 512), (255, 0, 0), 5)

## 画矩形
# 画一个绿色边框的矩形，参数2：左上角坐标，参数3：右下角坐标
cv2.rectangle(img, (384, 0), (510, 128), (0, 255, 0), 3)

## 画圆
# 画一个填充红色的圆，参数2：圆心坐标，参数3：半径 ,-1表示线宽，即填充
cv2.circle(img, (447, 63), 63, (0, 0, 255), -1)

## 画椭圆
# 参数2：椭圆中心(x,y) 参数3：x/y轴的长度 参数4：angle—椭圆的旋转角度 参数5：startAngle—椭圆的起始角度 参数6：endAngle—椭圆的结束角度
cv2.ellipse(img, (256, 256), (100, 50), 0, 0, 180, (255, 0, 0), -1)

## 画多边形
# 定义四个顶点坐标
pts = np.array([[10, 5], [50, 10], [70, 20], [20, 30]], np.int32)

# 顶点个数：4，矩阵变成4*1*2维
pts = pts.reshape((-1, 1, 2))
cv2.polylines(img, [pts], True, (0, 255, 255))

# 使用cv2.polylines()画多条直线
line1 = np.array([[100, 20], [300, 20]], np.int32).reshape((-1, 1, 2))
line2 = np.array([[100, 60], [300, 60]], np.int32).reshape((-1, 1, 2))
line3 = np.array([[100, 100], [300, 100]], np.int32).reshape((-1, 1, 2))
cv2.polylines(img, [line1, line2, line3], True, (0, 255, 255))

## 添加文字
# 参数2：要添加的文本 参数3：文字的起始坐标（左下角为起点） 参数4：字体 参数5：文字大小（缩放比例）
font = cv2.FONT_HERSHEY_COMPLEX
cv2.putText(img, 'ex2tron', (10, 500), font, 4, (255, 255, 255), 2, lineType=cv2.LINE_AA)

cv2.imshow('img', img)
cv2.waitKey(0)
