'''
使用固定阈值，自适应阈值和 Otus 阈值发 '二值化' 图像
opencv 函数 cv2.threshold(), cv2.adaptiveThreshold()
'''

## 固定阈值分割
# 像素点值大于阈值变成一类值，小于阈值变成另一类值
import cv2

# 灰度图读入
img = cv2.imread('gradient.jpg', 0)

# 阈值分割
# threshold 参数1：要处理的原图，一般是灰度图；参数2：设定的阈值；3：255；阈值方式
ret, th = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
cv2.imshow('threshold', th)
cv2.waitKey(0)
# print(ret)   127

import matplotlib.pyplot as plt

# 应用5种不同的阈值方法
ret, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
ret, th2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
ret, th3 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
ret, th4 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
ret, th5 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)

titles = ['Original', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
images = [img, th1, th2, th3, th4, th5]

# 使用Matplotlib显示
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(images[i], 'gray')
    plt.title(titles[i], fontsize=8)
    plt.xticks([]), plt.yticks([])  # 隐藏坐标轴
plt.show()

## 自适应阈值
# 固定阈值是在整幅图像上应用一个阈值进行分割，它并不适用于明暗分布不均的图片
# cv2.adaptiveThreshold() 自适应阈值每次取图片的一小部分计算阈值，这样图片不同区域的阈值就不尽相同。
# 自适应阈值对比固定阈值
img = cv2.imread('sudoku.jpg')

# 固定阈值
ret, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# 自适应阈值
'''
参数1：要处理的原图
参数2：最大阈值，一般为 255
参数3：小区域阈值的计算方式
ADAPTIVE_THRESH_MEAN_C：小区域内取均值
ADAPTIVE_THRESH_GAUSSIAN_C：小区域内加权求和，权重是个高斯核
参数4：阈值方式（跟前面讲的那5种相同）
参数5：小区域的面积，如 11 就是 11*11 的小块
参数6：最终阈值等于小区域计算出的阈值再减去此值
'''
th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 4)
th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 17, 6)

titles = ['Original', 'Global(v=127)', 'Adaptive Mean', 'Adaptive Gaussian']
images = [img, th1, th2, th3]

for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.imshow(images[i], 'gray')
    plt.title(titles[i], fontsize=8)
    plt.xticks([])
    plt.yticks([])
plt.show()
