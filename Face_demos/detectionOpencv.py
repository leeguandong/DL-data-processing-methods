'''
技术实现思路：
图片转成灰色（降低一维的灰度，减低计算强度）
脸上画矩形
使用训练分类器查找人脸
'''

# 图片转换成灰色，使用 opencv 的 cvtcolor 转换图片色彩
import cv2

filepath = 'F:/Github/DL-data-processing-methods/Demos/img/xingye-3.png'
img = cv2.imread(filepath)

# 转换灰度
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 显示图像
cv2.imshow('Image', gray)

cv2.waitKey(0)
cv2.destroyAllWindows()

# 图片上画矩阵，使用 opencv 的 rectangle() 绘制矩形
x = y = 10  # 坐标
w = 100  # 矩形大小（宽、高）
color = (0, 0, 255)  # 定义绘制颜色
cv2.rectangle(img, (x, y), (x + w, y + w), color, 1)  # 绘制矩形
cv2.imshow('Image', img)  # 显示图像
cv2.waitKey(0)
cv2.destroyAllWindows()

# 使用训练分类器查找人脸
'''
使用训练好的人脸模型，使用 Opencv 提供好的人脸分类模型 xml。
基于 Opencv 的人脸识别的算法研究与实现
(1) 高斯平滑技术。由于经常受到噪音的影响，会导致图像在进行预处理时发生数据损坏，图像细节受到损坏等，影响图像的整体质量。
在后期的提取，识别以及检测图像过程中都会受到一定的影响，因此需要去除噪音。在应用高斯平滑技术时，将其与自适应滤波法、中值
滤波滤波法等方式相比，无论是频率效果还是空间域，效果都更加明显，在增强低频信息的同时，能够将图像的边缘轮廓保留下来。
(2) 对比增强技术。应用对比度增强技术能够将图像中不同亮点进行分层，在测量过程中还能够统计各个点的像素，再将所得出的像素数据进行比较。
能进一步扩大各个点与周围像素之间的差异，从而增强对比度。在这一过程中，可以选择局部标准差，局部均值等方式进行比较和强化。
(3) Harr特征算法。

检测人脸
人脸识别系统，就是通过摄像头对人脸的轮廓和特征进行识别，其基础技术为轮廓对称检测方法，它可以在较为复杂的环境中获得关于人脸的信息，最终
筛选出更完整的人脸图像。

基于摄像头，在动态图像中获取一张静态的照片，同时锁定具体的位置，然后采用逐级处理技术对图片进行处理，同时以 Harr-like 为基础提取脸部信息，最后
将提取的脸部信息与数据库的信息进行对比，使用 Opencv 归一化平方差的匹配，完成人脸识别。

# 图像预处理
高斯平滑处理
灰度变换
对比度增强
二值化
# 识别特征
Harr-like 特征方式。目前，人脸检测的方式包括统计、知识，Harr-like 属于前者，Haar包括对角线特征、中心特征、线性特征、边缘特征四部分，作为判定
的基本元素，其特征图模块由白色和黑色构成，例如：某模板的特征值为F，其数值是有白色矩阵和黑色矩阵像素之差而获得的。在别是人脸特征的过程中，系统会
将图片划分为若干个小窗口，计算机根据各个区域所对应的特征，使用分类器对特征进行详细的筛选，从而做出更加准确的判断。在分析识别的过程中，系统可以
根据窗口设置的类型和矩阵位置与大小，从对应的图像中获取有价值的信息集合。
例如在一个尺寸为 24*24 像素的窗口中，就具有上万个以上的矩阵特征。
'''

# Opencv 人脸识别分类器
classifier = cv2.CascadeClassifier(
    'D:\python\python 3.5\Lib\site-packages\opencv-master\data\haarcascades\haarcascade_frontalface_default.xml')
color = (0, 255, 0)  # 定义绘制颜色

# 调用识别人脸
'''
gray:转换的灰图
scaleFactor:图像缩放比例，可理解为相机的 X 倍镜
minNeighbors：对特征检测点周边多少有效点同时检测，这样可避免因选择的特征检测点太小而导致遗漏
minSize:特征检测点的最小尺寸
'''
faceRects = classifier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))

# 传统算法对于眼睛和嘴巴的绘制都是经验式的
if len(faceRects):  # 大于 0 则检测出人脸
    for faceRect in faceRects:  # 单独框出每一张人脸
        x, y, w, h = faceRect
        # 框出人脸
        cv2.rectangle(img, (x, y), (x + h, y + w), color, 2)
        # 左眼
        cv2.circle(img, (x + w // 4, y + h // 4 + 30), min(w // 8, h // 8), color)
        # 右眼
        cv2.circle(img, (x + 3 * w // 4, y + h // 4 + 30), min(w // 8, h // 8), color)
        # 嘴巴
        cv2.rectangle(img, (x + 3 * w // 8, y + 3 * h // 4), (x + 5 * w // 8, y + 7 * h // 8), color)

cv2.imshow('image', img)  # 显示图像
c = cv2.waitKey(10)

cv2.waitKey(0)
cv2.destroyAllWindows()
