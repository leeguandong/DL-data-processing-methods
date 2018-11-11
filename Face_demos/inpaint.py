'''
图片修复原理
Opnecv 中，开发者标定噪声的特征，在使用噪声周围的颜色特征推理出应该修复的图片的颜色，从而实现实现图片修复
'''

'''
程序实现解析
1.标定噪声的特征，使用cv2.inRange二值化表示噪声对图片进行二值化处理，具体代码：cv2.inRange(img,np.array([240,240,240])),
np.array([255,255,255])),把 [240,240,240]-[255,255,255]以外的颜色处理为0.
2.使用 Opencv 的 dilate 方法，扩展特征的区域，优化图片处理效果。
3.使用 inpaint 方法，把噪声的 mask 作为参数，推理并修复图片。
'''

# 图片修复
import cv2
import numpy as np

path = 'F:/Github/DL-data-processing-methods/Face_demos/img/face_recognition/wa.jpg'

img = cv2.imread(path)
hight, width, depth = img.shape[0:3]

# 图片二值化处理，把[240,240,240]-[255,255,255]以外的颜色变成0
thresh = cv2.inRange(img, np.array([240, 240, 240]), np.array([255, 255, 255]))

# 创建形状和尺寸的结构元素
kernel = np.ones((3, 3), np.uint8)

# 扩张待修复区域
hi_mask = cv2.dilate(thresh, kernel, iterations=1)
specular = cv2.inpaint(img, hi_mask, 5, flags=cv2.INPAINT_TELEA)

cv2.namedWindow('Image', 0)
cv2.resizeWindow('Image', int(width / 2), int(hight / 2))
cv2.imshow('Image', img)

cv2.namedWindow('newImage', 0)
cv2.resizeWindow('newImage', int(width / 2), int(hight / 2))
cv2.imshow('newImage', specular)
cv2.waitKey(0)
cv2.destroyAllWindows()
