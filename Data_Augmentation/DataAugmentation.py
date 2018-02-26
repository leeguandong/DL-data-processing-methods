# 固定的数据探索与增强手段
import matplotlib.pyplot as  plt
import glob
from PIL import Image
from keras.preprocessing import image
import numpy as np

path = 'train/'
gen_path = 'result/'

def print_result(path):
    name_list = glob.glob(path)
    fig = plt.figure()
    for i in range(9):
        img = Image.open(name_list[i])
        # add_subplot(331) 参数一：子图总行数，参数二：子图总列数，参数三：子图位置
        sub_img = fig.add_subplot(331 + i)
        sub_img.imshow(img)
    plt.show()
    return fig

#####################################################################
# 打印图片列表
name_list = glob.glob(path + '*/*')
print(name_list)
# ['train\\00a366d4b4a9bbb6c8a63126697b7656.jpg', 'train\\00f34ac0a16ef43e6fd1de49a26081ce.jpg', 'train\\0a5f744c5077ad8f8d580081ba599ff5.jpg', 'train\\0a70f64352edfef4c82c22015f0e3a20.jpg', 'train\\0a783538d5f3aaf017b435ddf14cc5c2.jpg', 'train\\0a896d2b3af617df543787b571e439d8.jpg', 'train\\0abdda879bb143b19e3c480279541915.jpg', 'train\\0ac12f840df2b15d46622e244501a88c.jpg', 'train\\0b6c5bc46b7a0e29cddfa45b0b786d09.jpg']

# 打印图片
fig = print_result(path + '*/*')

# 保存图片
fig.savefig(gen_path + '/original_0.png', dpi=200, papertype='a5')

# #######################################################
# # 从文件夹中提取图片，并对原图进行归一化操作
datagen = image.ImageDataGenerator()
gen_data = datagen.flow_from_directory(path, batch_size=1, shuffle=False, save_to_dir=gen_path,
                                       save_prefix='dog_gen', target_size=(224, 224))
for i in range(9):
    gen_data.next()

fig = print_result(gen_path + '/*')
fig.savefig(gen_path + '/original_1.png', dpi=200, papertype='a5')

########################################################
# feature_wise 对整个数据集进行归一化操作
datagen = image.ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)

gen = image.ImageDataGenerator()

data = gen.flow_from_directory(path, batch_size=1, class_mode=None, shuffle=True, target_size=(224, 224))
np_data = np.concatenate([data.next() for i in range(data.n)])
# print(data.n)   9

datagen.fit(np_data)
'''
datagen有很多方法，fit只用于featurewise_center, featurewise_std_normalization and zca_whitening
Fits internal statistics to some sample data.
Required for featurewise_center, featurewise_std_normalization and zca_whitening.
'''

gen_data = datagen.flow_from_directory(path, batch_size=1, shuffle=False, save_to_dir=gen_path + '1', save_prefix='gen',
                                       target_size=(224, 224))

for i in range(9):
    gen_data.next()

fig = print_result(gen_path + '1/*')

fig.savefig(gen_path + '1/featurewise.png', dpi=200)

#####################################################################
# sample_wise 对输入数据每个样本本身做归一化，在一张图片上做归一化处理
datagen = image.ImageDataGenerator(samplewise_center=True, samplewise_std_normalization=True)

gen = image.ImageDataGenerator()

data = gen.flow_from_directory(path, batch_size=1, class_mode=None, shuffle=True, target_size=(224, 224))
np_data = np.concatenate([data.next() for i in range(data.n)])

datagen.fit(np_data)

gen_data = datagen.flow_from_directory(path, batch_size=1, shuffle=False, save_to_dir=gen_path + '2', save_prefix='gen',
                                       target_size=(224, 224))

for i in range(9):
    gen_data.next()

fig = print_result(gen_path + '2/*')

fig.savefig(gen_path + '2/samplewise.png', dpi=200)

#################################################################################
# zca_whtening白化
datagen = image.ImageDataGenerator(zca_whitening=True)

gen = image.ImageDataGenerator()

data = gen.flow_from_directory(path, batch_size=1, class_mode=None, shuffle=True, target_size=(36, 36))
np_data = np.concatenate([data.next() for i in range(1)])

datagen.fit(np_data)

gen_data = datagen.flow_from_directory(path, batch_size=1, shuffle=False, save_to_dir=gen_path + '3', save_prefix='gen',
                                       target_size=(36, 36))

for i in range(9):
    gen_data.next()

fig = print_result(gen_path + '3/*')

fig.savefig(gen_path + '3/zcawhtening.png', dpi=200)

###############################################################################
# rotation range 随机旋转
datagen = image.ImageDataGenerator(rotation_range=30)
gen_data = datagen.flow_from_directory(path, batch_size=1, shuffle=False, save_to_dir=gen_path + '4', save_prefix='gen',
                                       target_size=(224, 224))
for i in range(9):
    gen_data.next()

fig = print_result(gen_path + '4/*')
fig.savefig(gen_path + '4/rotation.png', dpi=200)

#############################################################################
# width_shift_range & height_shift_range 水平或竖直平移
datagen = image.ImageDataGenerator(width_shift_range=0.5, height_shift_range=0.5)

gen_data = datagen.flow_from_directory(path, batch_size=1, shuffle=False, save_to_dir=gen_path + '5', save_prefix='gen',
                                       target_size=(224, 224))

for i in range(9):
    gen_data.next()

fig = print_result(gen_path + '5/*')
fig.savefig(gen_path + '5/shift_range.png')

# # 一旦水平或者竖直平移超出区域，可以采用填充方法
datagen = image.ImageDataGenerator(width_shift_range=1, height_shift_range=1)
gen_data = datagen.flow_from_directory(path, batch_size=1, shuffle=False, save_to_dir=gen_path + '6', save_prefix='gen',
                                       target_size=(224, 224))

for i in range(9):
    gen_data.next()

fig = print_result(gen_path + '6/*')
fig.savefig(gen_path + '6/shift_range.png')

############################################################
# shear_range 错切坐标轴
datagen = image.ImageDataGenerator(shear_range=5)

gen_data = datagen.flow_from_directory(path, batch_size=1, shuffle=False, save_to_dir=gen_path + '7', save_prefix='gen',
                                       target_size=(224, 224))

for i in range(9):
    gen_data.next()

fig = print_result(gen_path + '7/*')
fig.savefig(gen_path + '7/shear_range.png')

################################################################
# zoom_range 在长或者宽的方向上进行放缩
datagen = image.ImageDataGenerator(zoom_range=0.2)
datagen = image.ImageDataGenerator(zoom_range=[0.5, 4])

gen_data = datagen.flow_from_directory(path, batch_size=1, shuffle=False, save_to_dir=gen_path + '8', save_prefix='gen',
                                       target_size=(224, 224))

for i in range(9):
    gen_data.next()

fig = print_result(gen_path + '8/*')
fig.savefig(gen_path + '8/zoom_range.png')

#####################################################################
# channel_shift_range 改变颜色通道数改变图片风格
datagen = image.ImageDataGenerator()

gen_data = datagen.flow_from_directory(path, batch_size=1, shuffle=False, save_to_dir=gen_path + '9', save_prefix='gen',
                                       target_size=(224, 224))

for i in range(9):
    gen_data.next()

fig = print_result(gen_path + '9/*')
fig.savefig(gen_path + '9/channel_shift_range.png')

###################################################################
# horizontal_flip & vertical_flip  水平或竖直翻转
datagen = image.ImageDataGenerator(horizontal_flip=True)

gen_data = datagen.flow_from_directory(path, batch_size=1, shuffle=False, save_to_dir=gen_path + '10',
                                       save_prefix='gen', target_size=(224, 224))

for i in range(9):
    gen_data.next()

fig = print_result(gen_path + '10/*')
fig.savefig(gen_path + '10/horizontal_flip.png')

###################################################################
# rescale 进行尺寸缩放
datagen = image.ImageDataGenerator(rescale=1 / 255)

gen_data = datagen.flow_from_directory(path, batch_size=1, shuffle=False, save_to_dir=gen_path + '11',
                                       save_prefix='gen', target_size=(224, 224))

for i in range(9):
    gen_data.next()

fig = print_result(gen_path + '11/*')
fig.savefig(gen_path + '11/rescale.png')
