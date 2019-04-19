import cv2
import numpy as np

# 开始计时
start = cv2.getCPUTickCount()

# 读入一张图片并调整对比度和亮度
img = cv2.imread('test1.jpg')
res = np.uint8(np.clip((0.8 * img + 80), 0, 255))

# 停止计时
end = cv2.getTickCount()

print((end - start) / cv2.getTickFrequency())
