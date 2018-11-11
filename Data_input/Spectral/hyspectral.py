import scipy.io as sio
import matplotlib.pyplot as plt
import spectral
import numpy as np
import cv2

# 字典，取数据段
mat_data = sio.loadmat('./datasets/IN/Indian_pines_corrected.mat')
data_IN = mat_data['indian_pines_corrected']

# 标签数据
mat_gt = sio.loadmat('./datasets/IN/Indian_pines_gt.mat')
gt_IN = mat_gt['indian_pines_gt']

# Scale the input between [0,1]
data_IN = data_IN.astype('float32')
data_IN -= np.min(data_IN)
data_IN /= np.max(data_IN)

# 均值滤波
img_mean = cv2.blur(data_IN, (5, 5))

# 高斯滤波
img_Guassian = cv2.GaussianBlur(data_IN, (5, 5), 0)

# 中值滤波
img_median = cv2.medianBlur(data_IN, 5)

# 双边滤波
# 9---滤波领域直径
# 后面两个数字：空间高斯函数标准差，灰度值相似性标准差
# data_IN.convertTo(data_IN, cv2.CV_32FC3, 1.0 / 255.0)
# cv2.error: OpenCV(3.4.3) D:\Build\OpenCV\opencv-3.4.3\modules\imgproc\src\smooth.cpp:5809: error: (-215:Assertion failed) (src.type() == CV_32FC1 || src.type() == CV_32FC3) && src.data != dst.data in function 'cv::bilateralFilter_32f'
img_bilater = cv2.bilateralFilter(data_IN, 9, 75, 75)

# 展示不同的图片
titles = ['srcImg', 'mean', 'Gaussian', 'median', 'bilateral']

imgs = [data_IN, img_mean, img_Guassian, img_median, img_bilater]

for i in range(5):
    # plt.subplot(2, 3, i + 1)
    spectral.imshow(imgs[i])
    plt.savefig('image' + str(i) + '.png')
    # plt.imshow(imgs[i])
    # plt.title(titles[i])

plt.show()

# input_image = spectral.imshow(data_IN)
# plt.savefig('image.png')
#
# ground_truth = spectral.imshow(classes=gt_IN)
# plt.savefig('gt.png')
