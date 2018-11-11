from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

img = Image.open('airplane00.jpg')
img = np.array(img)
# print(img)  数组
plt.imshow(img)
# plt.show()

# Flipping images with Numpy  翻转
flipped_img = np.fliplr(img)
plt.imshow(flipped_img)
# plt.show()

# shifting left
Height = 250
Width = 250
Depth = 3
for i in range(Height, 1, -1):
    for j in range(Width):
        if i < Height - 20:
            img[j][i] = img[j][i - 20]
        # 向左推进去这部分就自然消掉了
        elif i < Height - 1:
            img[j][i] = 0
plt.imshow(img)
# plt.show()

# shifting right
for j in range(Width):
    for i in range(Height):
        if j < Width - 20:
            img[i][j] = img[i][j + 20]
        else:
            img[i][j] = 0
plt.imshow(img)
# plt.show()

# shifting up

# shifting down

# 噪点
# 噪点是一种有趣的增强技术。 我已经看过很多关于对抗训练的有趣论文，你可以将一些噪点投入到图像中，因此模型无法正确分类。
# 但我仍然在寻找产生比下图更好的噪点的方法。 添加噪点可能有助于光照畸变并使模型更加鲁棒
# 这里让边缘光照发生失真，让尺度不变的情况下特征转变（SIFT），从而能增加鲁棒性（SURF)，使得识别更加准确
noise = np.random.randint(5, size=(250, 250, 3), dtype='uint8')

for i in range(Width):
    for j in range(Height):
        for k in range(Depth):
            if img[i][j][k] != 255:
                img[i][j][k] += noise[i][j][k]
plt.imshow(img)
plt.show()
