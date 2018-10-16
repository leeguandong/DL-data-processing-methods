import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

# print(tf.__version__)
# use Fashion MNIST dataset

fashion_mnist = keras.datasets.fashion_mnist

(train_images,train_labels),(test_images,test_labels) = fashion_mnist.load_data()
# 28*28 numpy arrays with pixel values ranging between 0 and 255. The labels are array of integers,ranging from 0 to 9

# explore the data
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)

# scale
train_images = train_images / 255.0
test_images = test_images / 255.0

# show 20 first image to correct format
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5+i)
    plt.xlabel([])
    plt.ylabel([])
    plt.grid(False)
    # plt.imshow(train_images[i],cmap = plt.cm.binary)
    # plt.xlabel()

# Build the model
model  = keras.Sequential([
    # no parameters to learn,just transform format of image
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128,activation=tf.nn.relu),
    keras.layers.Dense(10,activation=tf.nn.softmax)
])

# compile the model
'''
before the model is ready for training, it needs a few more setting
loss function
optimizer
metrics
'''
model.compile(optimizer = tf.train.AdamOptimizer(),loss = 'sparse_categorical_crossentopy',metrics=['accuracy'])

# Train the model
'''
Feed
assocaiate images and labels
make predictions about a test set
'''
model.fit(train_images,train_labels,epochs=5)


# Evaluate accuracy
# when acc on the test datset is a little less than the accuracy on the training dataset.the gap between training
# accuracy and test accuracy is an example of overfitting
test_loss,test_acc = model.evaluate(test_images,test_labels)

# Make predictions
predictions = model.predict(test_images)

# We can graph this to look at the full set of 10 channels
def plot_image(i,predictions_array,true_label,img):
    predictions










