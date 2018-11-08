'''
1.Build a model
2.Train this model on example
3.Use the model to make predictions about unkown data

已经将特征处理成四个维度了，采用 tf 进行建模后送入分类器处理
'''

# Setup program
'''
eager execution makes tf evaluate operations immediately,returning concrete values instead of creating
computational graph that us execyted later.
'''

# The Iris classification problem

# Import and parse the training dataset

# Create a tf.data.Dataset
'''
use make_csv_dataset function to spare the data into a suitable format.
The make_csv_dataset function returns a tf.data.Dataset of (features,label) pairs, where features is a
dictionary:{'feature':value}

To simplify model building step,create a function to repackage the featues dictionary into a single array with
shape:(batch_size,num_features)

This function uses the tf.stack method which takes values from a list of tensors and creates a combined tensor at
the specified dimension

Then use the tf.data.Dataset.map method to pack the (feature,label) pair into the training dataset

The features element of the Dataset are now arrays with shape (batch_size,num_featues)
'''

# Select the type of model

# Create a model using keras

# Training the model
'''
Define the loss and gradient function

Create an optimizer
An optimizer applies the computed gradients to model's variables to minimize loss function.
By iteratively calculating the loss and gradient for each batch,we will adjust the model during training.
Gradually,the model will find the best combination of weights and bias to minimize loss.

Training loop
1.Iterate each epoch.An epoch is one pass through the dataset
2.Within an epoch,iterate over each example in the training Dataset grabbing its feature(x) and label(y)
3.Using the example's features,make a prediction and compare it with the label.Measure the inaccuracy of
the prediction and use that to calculate the model's loss and gradients.
4.Use an optimizer to update the model's variables
5.Keep track of some stats for visualization
6.Repeat for each epoch

Visualize loss function over time

'''

# Evaluate the model's effectiveness
'''
Setup the test dataset

Evaluate the model on the test dataset
'''

# Use the trained model to make predictions
'''
let's use the trained model to make some predictions on unlabeled examples
'''



