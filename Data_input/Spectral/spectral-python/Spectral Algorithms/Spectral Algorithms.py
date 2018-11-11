from spectral import *

img = open_image('92AV3C.lan').load()

# Unsupervised Classification
# 无监督分类算法基于像素的光谱相似性将图像像素划分成组，而不使用光谱类的任何先验知识
# 20簇，30轮之后相似的聚集在一起
# k-means

# (tensorflow_gpu) C:\Users\KUN>e:
# (tensorflow_gpu) E:\>cd E:\tricks\spectral-python\Spectral Algorithms
# (tensorflow_gpu) E:\tricks\spectral-python\Spectral Algorithms>ipython
# Python 3.6.5 |Anaconda custom (64-bit)| (default, Mar 29 2018, 13:32:41) [MSC v.1900 64 bit (AMD64)]
# Type 'copyright', 'credits' or 'license' for more information
# IPython 6.4.0 -- An enhanced Interactive Python. Type '?' for help.

In [1]: from spectral import *

In [2]: img = open_image('92AV3C.lan').load()
C:\Users\KUN\Anaconda3\envs\tensorflow_gpu\Scripts\ipython:1: DeprecationWarning: tostring() is deprecated. Use tobytes() instead.

In [3]: (m,c) = kmeans(img,20,30)
# Initializing clusters along diagonal of N-dimensional bounding box.
# Iteration 1...21024 pixels reassigned.
# Iteration 2...11214 pixels reassigned.
# Iteration 3...4726 pixels reassigned.
# Iteration 4...1767 pixels reassigned.
# Iteration 5...1240 pixels reassigned.
# Iteration 6...1420 pixels reassigned.
# Iteration 7...1442 pixels reassigned.
# Iteration 8...1205 pixels reassigned.
# Iteration 9...1041 pixels reassigned.
# Iteration 10...934 pixels reassigned.
# Iteration 11...977 pixels reassigned.
# Iteration 12...1027 pixels reassigned.
# Iteration 13...1019 pixels reassigned.
# Iteration 14...1011 pixels reassigned.
# Iteration 15...904 pixels reassigned.
# Iteration 16...702 pixels reassigned.
# Iteration 17...653 pixels reassigned.
# Iteration 18...579 pixels reassigned.
# Iteration 19...518 pixels reassigned.
# Iteration 20...544 pixels reassigned.
# Iteration 21...554 pixels reassigned.
# Iteration 22...548 pixels reassigned.
# Iteration 23...548 pixels reassigned.
# Iteration 24...475 pixels reassigned.
# Iteration 25...418 pixels reassigned.
# Iteration 26...348 pixels reassigned.
# Iteration 27...327 pixels reassigned.
# Iteration 28...248 pixels reassigned.
# Iteration 29...215 pixels reassigned.
# Iteration 30...241 pixels reassigned.
# kmeans terminated with 17 clusters after 30 iterations.

In [4]: imshow(img)
# Out[4]:
# ImageView object:
#   Display bands       :  [0, 110, 219]
#   Interpolation       :  <default>
#   RGB data limits     :
#     R: [2632.0, 4536.0]
#     G: [1017.0, 1159.0]
#     B: [980.0, 1034.0]

In [5]: import pylab

In [6]: pylab.figure()
Out[6]: <Figure size 640x480 with 0 Axes>

In [7]: pylab.hold(1)
# C:\Users\KUN\Anaconda3\envs\tensorflow_gpu\Scripts\ipython:1: MatplotlibDeprecationWarning: pyplot.hold is deprecated.
#     Future behavior will be consistent with the long-time default:
#     plot commands add elements without first clearing the
#     Axes and/or Figure.
# C:\Users\KUN\Anaconda3\envs\tensorflow_gpu\lib\site-packages\matplotlib\__init__.py:911: MatplotlibDeprecationWarning: axes.hold is deprecated. Please remove it from your matplotlibrc and/or style files.
#   mplDeprecation)
# C:\Users\KUN\Anaconda3\envs\tensorflow_gpu\lib\site-packages\matplotlib\rcsetup.py:156: MatplotlibDeprecationWarning: axes.hold is deprecated, will be removed in 3.0
#   mplDeprecation)

In [9]: for i in range(c.shape[0]):
   ...:     pylab.plot(c[i])

In [10]: pylab.show()

# Supervised Classification
# Training Data
# 监督学习必须提供有标签的训练数据
In [12]: gt = open_image('92AV3GT.GIS').read_band(0)

In [13]: v = imshow(classes=gt)

# 需要产生训练数据，但是这里是把所有的数据都输入训练，之后用分类器对所有输入的数据在进行分类，这和深度学习是和区别的
In [14]: classes = create_training_classes(img,gt)

In [15]: gmlc = GaussianClassifier(classes)
# Setting min samples to 220
#   Omitting class   1 : only 54 samples present
#   Omitting class   7 : only 26 samples present
#   Omitting class   9 : only 20 samples present
#   Omitting class  13 : only 212 samples present
#   Omitting class  16 : only 95 samples present

In [16]: clmap = gmlc.classify_image(img)
Processing...done

In [17]: v= imshow(classes=clmap)

# 排除其他无关像素的干扰
In [18]: gtresults = clmap*(gt!=0)

In [19]: v = imshow(classes=gtresults)

# 查看分类错误的5类像素，原因是这5类像素的样本数据集太少了
In [21]: gtresults = gtresults*(gtresults != gt)

In [22]: v = imshow(classes=gtresults)

# Dimensionslity Reduction
# Processing hyperspectral images with hundreds of bands can be computationally burdensome and classification accuracy may suffer due to the so-called
#  “curse of dimensionality”. To mitigate these problems, it is often desirable to reduce the dimensionality of the data.

# Principal Components
# 高光谱图像中许多波段通常是高相关的。主成分变换将原始图像频带变成一组新的不相关的特征频带。这些新特征对应于图像协方差矩阵的特征向量，
# 可以在相对少量的主成分中捕获非常大百分比的图像方差（与原始频带数相比）
pc = principal_components(img)

In [24]: v = imshow(pc.cov)

# 为了使用主成分减少位数，我们可以按降序对特征值进行排序，然后保留足够的特征值（相应的特征向量）来捕获总图像的所需部分，然后，我们通过
# 将图像像素投影到剩余的特征向量上来减少图像像素的维度。我们将选择保留总图像差异的至少99.9%
In [25]: pc_0999 = pc.reduce(fraction=0.999)

In [26]: len(pc_0999.eigenvalues)
Out[26]: 32

In [27]: img_pc = pc_0999.transform(img)

In [28]: v = imshow(img_pc[:,:,:3],stretch_all=True)

In [29]: classes = create_training_classes(img_pc,gt)

In [30]: gmlc = GaussianClassifier(classes)
# Setting min samples to 32
#   Omitting class   7 : only 26 samples present
#   Omitting class   9 : only 20 samples present

In [31]: clmap = gmlc.classify_image(img_pc)
Processing...done

In [32]: clmap_training = clmap * (gt != 0 )

In [33]: v = imshow(classes = clmap_training)

In [34]: training_error = clmap_training * (clmap_training != gt)

In [35]: v = imshow(classes=training_error)

# Fisher Linear Discriminant
# Fisher线性判别式试图找到一组变换轴，其最大化类之间的平均距离与每个类内的样本之间的平均距离的比率。
 classes = create_training_classes(img,gt)

In [37]: fld = linear_discriminant(classes)

In [38]: len(fld.eigenvectors)
Out[38]: 220

In [39]: img_fld = fld.transform(img)

In [40]: v = imshow(img_fld[:,:,:3])

In [41]: classes.transform(fld.transform)

In [42]: gmlc = GaussianClassifier(classes)
# Setting min samples to 15

In [43]: clmap = gmlc.classify_image(img_fld)
Processing...done

In [44]: clmap_training = clmap * (gt != 0 )

In [45]: v = imshow(classes = clmap_training)

In [46]: fld_error = clmap_training * (clmap_training != gt)

In [47]: v = imshow(classes = fld_error)

# Target Detectors
# RX Anomaly Decetor
In [48]: rxvals = rx(img)

In [49]: from scipy.stats import chi2

In [50]: nbands = img.shape[-1]

In [51]: p = chi2.ppf(0.999,nbands)

In [52]: v = imshow(1*(rxvals>p))

In [53]: v = imshow(rxvals)

In [54]: import matplotlib.pyplot as plt

In [55]: f = plt.figure

In [56]: h = plt.hist(rxvals.ravel(),200,log=True)

# Miscellaneous Functions
# Band Resampling
# Comparing spectra measured with a particular sensor to spectra collected by a different sensor often requires resampling spectra to a common band discretization. Spectral bands of a single sensor may drift enough over time such that spectra collected by the same sensor at different dates requires resampling.

# NDVI
# 归一化差异植被指数

# Spectral Angle 光谱角




























