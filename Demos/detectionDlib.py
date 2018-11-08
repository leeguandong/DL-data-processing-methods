'''
比 Opencv 更加精准的图片人脸检测 Dlib 库
识别精准度： Dlib >= Opencv

Dlib 更多的人脸识别模型，可以检测脸部68个特征点甚至更多的特征点

训练模型用于是人脸识别的关键，用于查找图片的关键点
可以训练自己的人脸关键点识别，这个功能会放在后面讲。
'''

'''
人脸识别是一个比较复杂的过程，归纳起来可以由5个步骤组成：人脸检测、人脸关键点检测、人脸规整、人脸特征提取、人脸识别。
Dlib 库基于深度学习，利用已训练好的人脸关键点检测器和人脸识别模型，得到人脸面部特征值。
Dlib 库能够实现人脸检测和识别，其算法采用 HOG 特征和级联分类器，算法下面的大概过程如下：
(1) 将图像灰度化
(2) 采用 Gamma 校正法对图像进行颜色的标准化
(3) 对每个图像进行梯度计算
(4) 对图像进行小单元划分
(5) 生成每个单元格的梯度直方图
(6) 把单元格组合成大的块，块内归一化梯度直方图
(7) 生成 HOG 特征描述向量

实现过程：
根据人脸识别模型和人脸关键点检测器，得到已知人脸相片的特征值库
根据人脸识别模型和人脸关键点检测器，得到待识别人脸相片的特征值
计算待识别人脸相片的特征值和特征值库的欧式距离，距离最小者就是参考识别结果
'''
import cv2
import dlib

path = 'F:/Github/DL-data-processing-methods/Demos/img/gather.png'
img = cv2.imread(path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 人脸分类器
detector = dlib.get_frontal_face_detector()

# 获取人脸检测器
predictor = dlib.shape_predictor('D:\python\python 3.5\Lib\site-packages\dlib\shape_predictor_68_face_landmarks.dat\shape_predictor_68_face_landmarks.dat')

dets = detector(gray, 1)
for face in dets:
    shape = predictor(img, face)  # 寻找人脸的68个标定点
    # 遍历所有点，打印出其坐标，并圈出来
    for pt in shape.parts():
        pt_pos = (pt.x, pt.y)
        cv2.circle(img, pt_pos, 2, (0, 255, 0), 1)
    cv2.imshow('image', img)

cv2.waitKey(0)
cv2.destroyAllWindows()
