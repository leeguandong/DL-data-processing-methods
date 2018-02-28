from keras.layers import Input,Dense
from keras.models import Model
from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np

(data,labels),(x_test,y_test) = mnist.load_data()

x_train = data.reshape(len(data),-1)
y_train = np_utils.to_categorical(labels, 10)

# This returns a tensor
inputs = Input(shape=(784,))

# a layer instance is callable on a tensor, and returns a tensor
x = Dense(64, activation='relu')(inputs)
x = Dense(64, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)

# This creates a model that includes the Input layer and three Dense layers
model = Model(inputs=inputs, outputs=predictions)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train)  # starts training
model.summary()