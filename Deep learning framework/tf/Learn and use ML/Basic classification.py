import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from skimage import io, transform
import glob
import os

# import mnist_reader

# print(tf.__version__)
# use Fashion MNIST dataset

# train_images, train_labels = mnist_reader.load_mnist('/tf/Learn and use ML', kind='train')
# test_images, test_labels = mnist_reader.load_mnist('./', kind='t10k')

# fashion_mnist = keras.datasets.fashion_mnist
# (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
# 28*28 numpy arrays with pixel values ranging between 0 and 255. The labels are array of integers,ranging from 0 to 9

# from tensorflow.examples.tutorials.mnist import input_data

# data = input_data.read_data_sets('data/fashion')

# train_images, train_labels = data.train.images, data.train.labels
# test_images, test_labels = data.test.images, data.test.labels

# class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
#                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

class_names = ['argricultural', 'airplane', 'baseballdiamond', 'beach', 'buildings', 'chaparral', 'denseresidential',
               'forest', 'freeway', 'golfcourse', 'harbor', 'intersection', 'mediumresidential', 'mobilehomepark',
               'overpass', 'parkinglot', 'river', 'runway', 'sparseresidential', 'storagetanks',
               'tenniscourt']

path = './data/remote sensing/'

w = 224
h = 224


# Input images
def read_img(path):
    cate = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]
    imgs = []
    labels = []
    for idx, folder in enumerate(cate):
        for im in glob.glob(folder + '/*.jpg'):
            print('reading the images:%s' % (im))
            img = io.imread(im)
            img = transform.resize(img, (w, h))
            imgs.append(img)
            labels.append(idx)
    return np.asarray(imgs, np.float32), np.asarray(labels, np.int32)


data, label = read_img(path)
# print(data)

# shuffle
num_example = data.shape[0]
arr = np.arange(num_example)
np.random.shuffle(arr)
data = data[arr]
label = label[arr]

# set dataset into training data and validation data
ratio = 0.8
s = np.int(num_example * ratio)
train_images = data[:s]
train_labels = label[:s]
test_images = data[s:]
test_labels = label[s:]

# explore the data
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

# scale
train_images = train_images / 255.0
test_images = test_images / 255.0

# show 20 first image to correct format
# plt.figure(figsize=(10, 10))
# for i in range(25):
#     plt.subplot(5, 5, 1 + i)
#     plt.xlabel([])
#     plt.ylabel([])
#     plt.grid(False)
#     plt.show()
#     plt.imshow(train_images[i])
#     plt.xlabel(class_names[train_labels[i]])

# Build the model
model = keras.Sequential([
    # no parameters to learn,just transform format of image
    keras.layers.Flatten(input_shape=(224, 224, 3)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(21, activation=tf.nn.softmax)
])

# compile the model
'''
before the model is ready for training, it needs a few more setting
loss function
optimizer
metrics
'''
model.compile(optimizer=tf.train.AdamOptimizer(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
'''
Feed
assocaiate images and labels
make predictions about a test set
'''
model.fit(train_images, train_labels, epochs=5)

# Evaluate accuracy
# when acc on the test datset is a little less than the accuracy on the training dataset.the gap between training
# accuracy and test accuracy is an example of overfitting
test_loss, test_acc = model.evaluate(test_images, test_labels)

# Make predictions
predictions = model.predict(test_images)


# We can graph this to look at the full set of 10 channels
def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xlabel([])
    plt.ylabel([])

    plt.imshow(img)

    predictions_label = np.argmax(predictions_array)
    if predictions_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel('{}{:2.0f}%({})'.format(class_names[predictions_label], 100 * np.max(predictions_array),
                                       class_names[true_label]), color=color)
    plt.show()


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xlabel([])
    plt.ylabel([])
    thisplot = plt.bar(range(21), predictions_array)
    plt.ylim([0, 1])  # Get or set the *y*-limits of the current axes
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')
    plt.show()


# Let is look at the 0th image, predictions, and prediction array
i = 0
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions, test_labels)

i = 12
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions, test_labels)

# Plot the first X test images,their predicted label, and true label
num_rows = 5
num_cols = 3
num_images = num_rows * num_cols
plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    plot_image(i, predictions, test_labels, test_images)
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    plot_value_array(i, predictions, test_labels)

# Finally, use an image from the test dataset
img = test_images[0]
print(img.shape)

# tf.keras models are optimized to make predictions on batch,or collection, of example at once. So even
# though we are using a single image,we need to add it to a list
img = (np.expand_dims(img, 0))
print(img.shape)

# Now predict the image
predictions_single = model.predict(img)
print(predictions_single)

# return a list of lists,one for each image in the batch of data.Grab the predictions for our image in the batch
plot_value_array(0, predictions_single, test_labels)
_ = plt.xticks(range(21), class_names, rotation=45)

np.argmax(predictions_single[0])
