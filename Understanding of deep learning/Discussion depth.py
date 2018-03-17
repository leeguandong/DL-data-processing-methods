from keras import Model
from keras.layers import *

input = Input(shape=(11, 11, 200, 1))

conv1 = Conv3D(16, kernel_size=(3, 3, 20), strides=(1, 1, 10), padding='valid',
               kernel_regularizer=regularizers.l2(0.01))(input)
act1 = Activation('relu')(conv1)
conv2 = Conv3D(16, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same', kernel_regularizer=regularizers.l2(0.01))(
    act1)
act2 = Activation('relu')(conv2)
pool1 = MaxPooling3D(pool_size=(2, 2, 3))(act2)

conv3 = Conv3D(16, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same', kernel_regularizer=regularizers.l2(0.01))(
    pool1)
act3 = Activation('relu')(conv3)
conv4 = Conv3D(16, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same', kernel_regularizer=regularizers.l2(0.01))(
    act3)
act4 = Activation('relu')(conv4)
pool2 = MaxPooling3D(pool_size=(2, 2, 3))(act4)

conv5 = Conv3D(16, kernel_size=(2, 2, 2), strides=(1, 1, 1), padding='same', kernel_regularizer=regularizers.l2(0.01))(
    pool2)
act5 = Activation('relu')(conv5)
conv6 = Conv3D(16, kernel_size=(2, 2, 2), strides=(1, 1, 1), padding='same', kernel_regularizer=regularizers.l2(0.01))(
    act5)
act6 = Activation('relu')(conv6)
pool3 = MaxPooling3D(pool_size=(2, 2, 2))(act6)

# flatten1 = Flatten()(pool3)
# fc1 = Dense(128)(flatten1)
# act7 = Activation('relu')(fc1)
# drop1 = Dropout(0.5)(act7)
#
# dense = Dense(units=12, activation="softmax", kernel_initializer="he_normal")(drop1)

model = Model(inputs=input, outputs=pool3)
model.summary()

'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         (None, 11, 11, 200, 1)    0
_________________________________________________________________
conv3d_1 (Conv3D)            (None, 9, 9, 19, 16)      2896
_________________________________________________________________
activation_1 (Activation)    (None, 9, 9, 19, 16)      0
_________________________________________________________________
conv3d_2 (Conv3D)            (None, 9, 9, 19, 16)      6928
_________________________________________________________________
activation_2 (Activation)    (None, 9, 9, 19, 16)      0
_________________________________________________________________
max_pooling3d_1 (MaxPooling3 (None, 4, 4, 6, 16)       0
_________________________________________________________________
conv3d_3 (Conv3D)            (None, 4, 4, 6, 16)       6928
_________________________________________________________________
activation_3 (Activation)    (None, 4, 4, 6, 16)       0
_________________________________________________________________
conv3d_4 (Conv3D)            (None, 4, 4, 6, 16)       6928
_________________________________________________________________
activation_4 (Activation)    (None, 4, 4, 6, 16)       0
_________________________________________________________________
max_pooling3d_2 (MaxPooling3 (None, 2, 2, 2, 16)       0
_________________________________________________________________
conv3d_5 (Conv3D)            (None, 2, 2, 2, 16)       2064
_________________________________________________________________
activation_5 (Activation)    (None, 2, 2, 2, 16)       0
_________________________________________________________________
conv3d_6 (Conv3D)            (None, 2, 2, 2, 16)       2064
_________________________________________________________________
activation_6 (Activation)    (None, 2, 2, 2, 16)       0
_________________________________________________________________
max_pooling3d_3 (MaxPooling3 (None, 1, 1, 1, 16)       0
=================================================================
Total params: 27,808
Trainable params: 27,808
Non-trainable params: 0
_________________________________________________________________
'''
