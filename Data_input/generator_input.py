from keras.preprocessing import image
from keras.applications import *
from keras.preprocessing.image import ImageDataGenerator

train_path = ''
val_path = ''

# 这里先定义一个生成器
generator = ImageDataGenerator()

# 生成器从指定路径中生成图片
train_data = generator.flow_from_directory(train_path, batch_size=1)
val_data = generator.flow_from_directory(val_path, batch_size=1)

# 定义模型
model = ResNet50(weights='imagenet')
model.compile(optimizer= 'adam',loss='catagorical_crosscentropy',metrics=['accuracy']

              )

