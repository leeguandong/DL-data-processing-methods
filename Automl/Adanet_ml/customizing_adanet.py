'''
 this tutorial, we will explore the flexibility of the adanet framework, and create a custom search space for an
 image-classificatio dataset using high-level TensorFlow libraries like the tf.keras.layers functional API.
'''

# Fashion MNIST dataset

# Supply the data in Tensorflow
'''
Our first task is to supply the data in TensorFlow. Using the tf.estimator.Estimator covention, we will define a function
that returns an input_fn which returns feature and label Tensors.

We will also use the tf.data.Dataset API to feed the data into our models.
'''


