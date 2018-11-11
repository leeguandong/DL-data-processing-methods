# 调用GUI窗口，要在ipython中设置 ipython - pylab
# ipython -pylab = wx
from spectral import *
import matplotlib.pyplot as plt

img = open_image('92AV3C.lan')

# ( ) 选三个波段，展示出来的是RGB的形式
# print(imshow(img, (29, 19, 200)))
# plt.savefig('img0')
'''
ImageView object:
  Display bands       :  (29, 19, 200)
  Interpolation       :  <default>
  RGB data limits     :
    R: [2054.0, 6317.0]
    G: [2775.0, 7307.0]
    B: [1009.0, 1549.0]
'''

# Class Map Display
gt = open_image('92AV3GT.GIS').read_band(0)
view = imshow(classes=gt)
plt.savefig('gt')

# 显示模式
view = imshow(img, (30, 20, 10), classes=gt)
view.set_display_mode('overlay')
view.class_alpha = 0.5

# Interactive Class Labeling
# 交互式类标签

# Saving RGB Image Files
# save_rgb('rgb.jpg', img, [29, 19, 9])
# 调色板colors必须指定参数，否则是single-band灰度图展示
# save_rgb('gt2.jpg', gt, colors=spy_colors)

# Spectrum Plots
# 将光谱频带信息和图像关联起来,x轴显示波段
# 光谱图旨在快速查找图像中的光谱信息，如果想要更漂亮的数据图，可以使用Spy从图像中读取光谱，并直接使用matplotlib创建自定义图
import spectral.io.aviris as aviris

img.bands = aviris.read_aviris_bands('92AV3C.spc')

# Hypercube Dispaly
view_cube(img, bands=[19, 19, 9])

# N-Dimensional Feature Display
# 由于高光谱图像包含数百个窄的连续波段，因此波段（特别是相邻波段）之间通常存在强相关性，为了增加所显示的信息量，通常将图像的维度降低到具有较高信息密度的较小特征集上（例如，
# 主成分变换）。然而，在变换图像中通常仍然存在多余三个以上的特征，因此需要决定哪些最佳三个特征以突出数据集的某些方面（例如，光谱类别可分性）
# In most cases, the display will be more useful by first performing dimensionality reduction prior to viewint the data
# (e.g., by selecting some number of principal components)
data = open_image('92AV3C.lan').load()
gt = open_image('92AV3GT.GIS').read_band(0)
pc = principal_components(data)
xdata = pc.transform(data)
w = view_nd(xdata[:, :, :15], classes=gt)

# Display Modes
# Single-Octant
# Mirrored-Octant
# Independent-Octant
