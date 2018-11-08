# Options
'''
There are different ways to save tensorflow models-depending on the API you are using.
This guide uses tf.keras,a high-level API to build and train models in Tensorflow.
'''

# Get an example dataset
'''
We will use MNIST dataset to train our model to demonstrate saving weights.To speed up these
demonstration runs,only use the first 1000 examples.
'''

# Define a model

# Save checkpoints during training
'''
tf.keras.callback.ModelCheckpoint is a callback that performs this task.
The primary use case is to automatically save checkpoints during and at the end of training.

This creates a single collection of tf checkpoint files that are updated at the end of each epoch.

Create a new,untrained model.When restoring a model from only weights,you must have a model with
the same architecture as the original model.Since it's the same model architecture,we an share
weights despite that it's a different instance of the model.
'''

# Now rebuild a fresh,untrained mdoel,and evaluate it on the test data
# Load the weights from the checkpoint,and re-evaluate

# Checkpoint callback options
'''
The callback provides several options to give the resulting checkpoints unique names,and adjust the
checkpointing frequency.
'''

# What are these files
'''
The above code stores the weights to a collection of checkpoint-formatted files that contain only
the trained weights in a binary format,Checkpoint contain: One or more shards that contain your
model's weights.
'''

# Manually save weights
'''
Above you saw how to load the weights into a model
'''

# Save the entire model
'''
The entire model can be saved to file that contains the weight values,the model's configureation,
and even the optimizer's configuration.This allows you to chechpoint a model and resume training
later - from the exact same state - without access to the code.

keras provides a basic save format using the hdf5 standard. For our purposes,the saved model can be
treated as a single binary blob.

This technique saves everything
The weight values
The model's configuration
The optimizer configuration


'''



