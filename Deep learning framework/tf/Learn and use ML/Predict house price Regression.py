'''
In a regression problem,we aim to predict the output of a continuous value,like a price or a probability
Contrast this with a classification problem,where we aim to predict a discrete label.

To predict the median price of homes in a Boston suburb during the mid-1970s.To do this.we will provide
the model with some data points about the suburb,such as the crime rate and the local property ta rate
'''
from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow import keras

import numpy as np

# The Boston Housing Prices dataset
boston_housing = keras.datasets.boston_housing
(train_data, train_labels), (test_data, test_labels) = boston_housing.load_data()

# Shuffle the training set
order = np.argsort(np.random.random(train_labels.shape))
train_data = train_data[order]
train_labels = train_labels[order]

# Examples and feature
# 506 total examples, and 404 training examples and 102 test examples
print('Training set: {}'.format(train_data.shape))
print('Testing set: {}'.format(test_data.shape))

# 13 different features

# Use the pandas to display the first few rows of the dataset in a nicely formatted table
import pandas as pd

column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAO', 'TAX', 'PTRATIO', 'B', 'LSTAT']

df = pd.DataFrame(train_data, columns=column_names)
df.head()

# Labels
# The labels are the house prices in thousand of dollars

# Normalize features
'''
Use different scales and ranges. subtract the mean of the feature and divide by the standard deviation.
but, a point need to focus,we should make data to train,validation and test data first,then only cacaulate
mean,std from train data,and every data all less this mean.

Although the model might converge without feature normalization, it makes training more difficult, and
it makes the resulting model more dependent on the choice of units used in the input.
'''
mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
train_data = (train_data - mean) / std
test_data = (test_data)

print(train_data[0])


# create the model
def build_model():
    model = keras.Sequential([
        keras.layers.Dense(64, activation=tf.nn.relu, input_shape=(train_data.shape[1],)),
        keras.layers.Dense(64, activation=tf.nn.relu),
        keras.layers.Dense(1)
    ])

    optimizer = tf.train.RMSPropOptimizer(0.001)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])

    return model


model = build_model()
model.summary()


# Train the model
# The model is trained for 50 epoches,and record the training and validation accuracy in the histoty object
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0:
            print('')
        print('.', end='')


Epoches = 50

# Store training stats
history = model.fit(train_data, train_labels, epochs=Epoches, validation_split=0.2, verbose=0, callbacks=[PrintDot()])

'''
Visualize the model's training progress using the stats stored in the history object. We want to use this data
data to determine how long to train before the model stops making progress.
'''
import matplotlib.pyplot as plt


# mean_absolute_error cacaulate
def plot_history(history):
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [1000]')
    plt.plot(history.epoch, np.array(history.history['mean_absolute_error']), label='Train Loss')
    plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']), label='Val loss')
    plt.legend()
    plt.ylim([0, 5])
    plt.show()


plot_history(history)

'''
Very important method to avoid overfitting

above graph shows little improvement in the model after ablout xx epoch,Let's update the model.fit method to
automatically stop training when the validation score doesn't improve.We will use a callback that tests a
training condition for every epoch.If a set amount of epochs elapses without showing improvement,then automatically
stop the training
'''
model = build_model()

# The patience parameter is the amount of epoches to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

history = model.fit(train_data, train_labels, epochs=Epoches, validation_split=0.2, verbose=0,
                    callbacks=[early_stop, PrintDot()])
plot_history(history)

# model performance on the test
[loss, mae] = model.evaluate(test_data, test_labels, verbose=0)
print('Testing set Mean Abs Error: ${:7.2f}'.format(mae * 1000))

# predict
test_predictions = model.predict(test_data).flatten()

plt.scatter(test_labels, test_predictions)
plt.xlabel('True Value [1000$]')
plt.ylabel('Predictions [1000$]')
plt.axis('equal')
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())
_ = plt.plot([-100, 100], [-100, 100])

error = test_predictions - test_labels
plt.hist(error, bins=50)
plt.xlabel('Prediction Error [1000$]')
_ = plt.ylabel('Count')

# Conclusion
'''
The notebook introduced a few techniques to handle a regression problem

Mean Squared Error(MSE) is a common loss function used for regression problems(different than clasification problem)

Similarly, evaluation metrics uesd for regression differ from classification.A common regression metrics
is Mean Absolute Error(MAE)

When input data features have values with different ranges,each feature should be scaled independently.

If there is not much training data,prefer a small network with few hidden layer to avoid overfiting

Early stopping is useful technique to prevent overfitting
'''
