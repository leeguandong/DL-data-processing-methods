'''
The notebook classifiers movie reviews as positive or negative using the text of the review.
This is an example of binary or two-class-classification,an important and widely applicable kind of machine
learning problem

We use IMDB dataset

dataset should be integers,but we need convert to tensor to fed the network,also we explore full text of dataset
'''

import tensorflow as tf
from tensorflow import keras
import numpy as np

# Download the IMDB dataset
# 10000 most frequently occuring words in the training data and in our experience is 3908 size of dictionary
imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# Explore the data
# Each example is an array of integers representing the words of the movie review,Each label is an integer value of either 0 or 1
print('Training entries:{},labels:{}'.format(len(train_data), len(train_labels)))

# The text of reviews have been converted to integers,where each integer represents a specific word in a dictionary
print(train_data[0])

# Inputs to a a neural network must be the same length ,we will need to resolve this later
# Use pad_sequences
# Convert the integers back to words, It may be useful to know how to convert integers back to text.Here,
# we will create a helper function to query a dictionary object that contains the integer to string mapping
# A dictionary mapping words to an integer index
word_index = imdb.get_word_index()

# The first indices are reserved
word_index = {k: (v + 3) for k, v in word_index.items()}
word_index['<PAD>'] = 0
word_index['<START>'] = 1
word_index['<UNK>'] = 2  # unknown
word_index['UNUSED'] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


# Now we can use the decode_review function to display the text for the first review
decode_review(train_data[0])

# Prepare the data
'''
The reviews-the arrays of integers-must be converted to tensors before fed into the neural network.This conversion
can be done a couple of ways:

one-hot-encode the arrays to convert them into vectors of 0s and 1s. for example,the sequence[3,5] would be
become a 10000-dimensional vector that is all zeros except for indices 3 and 5, whilch are ones. Then,make
this the first layer in our network-Dense layer-that can handle floating point vector data.This approach is memory
intensive,though,requring a num_words*num_reviews size matrix.

Alternatively,we can pad the arrays so that all have the same length,then create integer tensor of shape
max_length*num_reviews.We can use an embedding layer capable of handling this shape as the first layer in our network
'''
# the movie review must be the same length,we will use the pad_sequences function to standarize the length
# all standrize to 256 size,and fill in 0 and what is digit not string
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index['<PAD>'], padding='post',
                                                        maxlen=256)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index['<PAD>'], padding='post', maxlen=256)

# Build the model
'''
two main architectural decisions:
    how many layers to use in the model
    how many hidden units to use for each layer
'''
vocab_size = 10000

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

model.summary()

'''
Embedding layer: This layer takes the integer-encoded vocabulary and looks up the embedding vector for each
word-index.These vectors are learned as the model trains. The vectors add a dimension to the output array
The resulting dimension are:(batch,sequence,embedding)

GlobalAveragePooling1D: return a fixed-length output vector for each example by averaging over the sequence
dimension.

'''

# Hidden units
'''
If a model has more hidden units(a higher-dimensional representation space), and more layers,then the network can
learn more complex representations.However,it makes the network more computationally expensive and may lead
to learning unwanted patterns-patterns that imporve performance on training data but not on the test data.
This is called overfitting, and we will explore it later.
'''

# Loss function and optimizer
'''
binary_crossentropy
mean_squared_error    regression problem(to predict the price of a house)
'''
model.compile(optimizer=tf.train.AdamOptimizer(), loss='binary_crossentropy', metrics=['accuracy'])

# Create a validation set
'''
When training,we want to check the accuracy of the model on data it has not seen before.create a validation set by setting
apart 10000 examples from the original training data.
'''
x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

# Train the model
'''
This is 40 iterations over all samples in the x_train and y_train tensor.Number of epochs to train the model.
An epoch is an iteration over the entire `x` and `y` data provided.
'''
history = model.fit(partial_x_train, partial_y_train, epochs=40, batch_size=512, validation_data=(x_val, y_val),
                    verbose=1)

# Evaluate the model
results = model.evaluate(test_data, test_labels)
print(results)
# [0.34018163004875185, 0.87144]

# Create a graph of accuracy and loss over time
# model.fit() return a History object that contains a dictionary with everything that happened during training
history_dict = history.history
history_dict.keys()

# There are four entries:one for each monitored metric during training and validation.We can use these to plot
import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epoches = range(1, len(acc) + 1)

# bo is for blue dot, b is for solid blue line
plt.plot(epoches, loss, 'bo', label='Training loss')
plt.plot(epoches, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epoches')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.clf()
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

plt.plot(epoches, acc, 'bo', label='Training acc')
plt.plot(epoches, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epoches')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
