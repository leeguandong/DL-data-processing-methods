import cv2

# 1.灰度图加载一张彩色图
img = cv2.imread('test1.jpg', 0)

# 2.显示图片
cv2.imshow('test1', img)
cv2.waitKey(0)

# 先定义窗口，后显示图片
cv2.namedWindow('test1', cv2.WINDOW_NORMAL)
cv2.imshow('test1', img)
cv2.waitKey(0)

# 3.保存图片
cv2.imwrite('test1_gray.jpg', img)
