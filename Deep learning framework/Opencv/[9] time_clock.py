import cv2
import math
import datetime
import numpy as np

# 上下左右边距
margin = 5
# 圆的半径
radius = 220
# 圆心
center = (center_x, center_y) = (255, 255)

# 1.新建一个画板并填充成白色
img = np.zeros((450, 450, 3), np.uint8)
img[:] = (255, 255, 255)

# 2. 画出圆盘
cv2.circle(img, center, radius, (0, 0, 0), thickness=5)
