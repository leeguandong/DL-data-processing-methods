# 1.高斯滤波器
import cv2
import numpy as np
import matplotlib.pyplot as plt

# img = cv2.imread('airplane00.jpg')
img = cv2.imread('forest12.jpg')

blur = cv2.GaussianBlur(img, (5, 5), 0)
plt.subplot(121), plt.imshow(img), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(blur), plt.title('Averaging')
plt.xticks([]), plt.yticks([])
# cv2.imwrite('gussfiltered.jpg', blur)
# plt.show()

# 2.均值滤波器
# 均值滤波器是一个简单的滑动窗口，用窗口中的所有像素值的平均值替代中心值。窗口或核通常
# 是正方形，但它可以是任何形状。
blur = cv2.blur(img, (5, 5))

plt.subplot(121), plt.imshow(img), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(blur), plt.title('Averaging')
plt.xticks([]), plt.yticks([])
# plt.show()
# plt.savefig('possionfiltered.jpg')

# 中值过滤器,效果是不错的
# 中心点想租被窗口中的中位数所代替
blur = cv2.medianBlur(img, 5)

plt.subplot(121), plt.imshow(img), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(blur), plt.title('Averaging')
plt.xticks([]), plt.yticks([])
# plt.show()

# 双边过滤器
# 双边滤波器使用高斯滤波，但他有一个乘法分量，它是像素强度差的函数。他确保在计算模糊强度值时仅包括与中心像素类似的像素强度。磁过滤器保留边缘。
blur = cv2.bilateralFilter(img, 9, 75, 75)

plt.subplot(121), plt.imshow(img), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(blur), plt.title('Averaging')
plt.xticks([]), plt.yticks([])
# plt.show()


###################################################################
# 均值滤波
img_mean = cv2.blur(img, (5, 5))

# 高斯滤波
img_Guassian = cv2.GaussianBlur(img, (5, 5), 0)

# 中值滤波
img_median = cv2.medianBlur(img, 5)

# 双边滤波
img_bilater = cv2.bilateralFilter(img, 9, 75, 75)

# 展示不同的图片
titles = ['srcImg', 'mean', 'Gaussian', 'median', 'bilateral']
imgs = [img, img_mean, img_Guassian, img_median, img_bilater]

for i in range(5):
    plt.subplot(2, 3, i + 1)
    plt.imshow(imgs[i])
    plt.title(titles[i])
plt.show()
