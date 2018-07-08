from keras import Model
from keras.layers import *

input = Input(shape=(11, 11, 200, 1))

conv1 = Conv3D(16, kernel_size=(3, 3, 20), strides=(1, 1, 1), padding='same',
               kernel_regularizer=regularizers.l2(0.01))(input)
act1 = Activation('relu')(conv1)
conv2 = Conv3D(16, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same', kernel_regularizer=regularizers.l2(0.01))(
    act1)
act2 = Activation('relu')(conv2)
# pool1 = MaxPooling3D(pool_size=(2, 2, 3))(act2)

bn_axis = 4 if K.image_data_format() == 'channels_last' else 1
x = Concatenate(axis=bn_axis)([input, act2])

conv3 = Conv3D(16, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same', kernel_regularizer=regularizers.l2(0.01))(
    x)
act3 = Activation('relu')(conv3)
conv4 = Conv3D(16, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same', kernel_regularizer=regularizers.l2(0.01))(
    act3)
act4 = Activation('relu')(conv4)
# pool2 = MaxPooling3D(pool_size=(2, 2, 3))(act4)

bn_axis = 4 if K.image_data_format() == 'channels_last' else 1
x = Concatenate(axis=bn_axis)([x, act4])

# flatten1 = Flatten()(pool3)
# fc1 = Dense(128)(flatten1)
# act7 = Activation('relu')(fc1)
# drop1 = Dropout(0.5)(act7)
#
# dense = Dense(units=12, activation="softmax", kernel_initializer="he_normal")(drop1)

model = Model(inputs=input, outputs=x)
model.summary(positions=[.33, .61, .71, 1.])
