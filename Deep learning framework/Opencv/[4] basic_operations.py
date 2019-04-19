'''
访问和修改图片像素点的值
获取图片的宽、高、通道数的属性
了解感兴趣区域 ROI
分离和合并图像通道
'''

## 获取和修改像素点值
import cv2

img = cv2.imread('test1.jpg')
cv2.imshow('img', img)

# 通过行列坐标来获取像素点的值
px = img[100, 90]
print(px)

# 只获取蓝色 blue 通道的值,前面是坐标
px_blue = img[100, 90, 0]
print(px_blue)

# 修改像素值
img[100, 90] = [255, 255, 255]
print(img[100, 90])

print(img.shape)
# 形状中包括行数、列数和通道数
height, width, channels = img.shape

print(img.dtype)

print(img.size)

## ROI
# 截取 ROI
area = img[100:200, 115:188]
cv2.imshow('area', area)
cv2.waitKey(0)

## 通道分割和合并
b, g, r = cv2.split(img)
img = cv2.merge((b, g, r))

# split()函数比较耗时，更高效的方式是用 numpy 中的索引
b = img[:, :, 0]
cv2.imshow('blue', b)
cv2.waitKey(0)
