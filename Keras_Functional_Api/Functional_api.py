from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
from keras.utils import np_utils
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

(data, labels), (x_test, y_test) = mnist.load_data()

x_train = data.reshape(len(data), -1)
y_train = np_utils.to_categorical(labels, 10)

inputs = Input(shape=(784,))

x = Dense(64, activation='relu')(inputs)
x = Dense(64, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)

# This creates a model that includes the Input layer and three Dense layers
model = Model(inputs=inputs, outputs=predictions)
model.summary()
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# starts training
model.fit(x_train, y_train)

