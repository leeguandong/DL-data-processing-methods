'''
This is an intorductory tutorial for using Tensorflow.It will cover.
Importing required packages
Creating and using Tensors
Using GPU acceleration
Datasets
'''

# To get started, eager execution

# Tensor
'''
A Tensor is a multi-dimensional array. Similar to numpy ndarray objects,Tensor objects have a data type
and a shape

Each Tensor has a shape and a datatype

The most obvious differences between Numpy arrays and Tensorflow Tensors are:
1.Tensors can be backed by accelerator memory(like GPU,TPU)
2.Tensors are immutable

Sharing the underlying representation isn't always possible since the Tensor may be hosted in GPU memory
while Numpy arrays are always backed by host memory,and the conversion will thus involve a copy from
GPU to host memory.
Tensor在Gpu内存中，但是numpy数组在内存中，所以高光谱那个问题有可能是内存问题，而不是GPU问题,两者之间的转化涉及的是主机内存
向GPU内存之间的转化问题
'''

# GPU acceleration
'''
Tensorflow automatically decides whether to use the GPU or CPU for an operation(and copies the tensor
between CPU and GPU memory if necessary).Tensors produced by an operation are typically backed by the memory
of which the operation executed.
'''

# Device Names
'''
The Tensor .device property provides a fully qualified string name of the device hosting the contents of the tensor.
The string ends with GPU:<N> if the tensor is placed on the N-th GPU on the host.

However,Tensorflow operations can be explicitly placed on specific devices using the tf.device content
manager.
'''

# Dataset
'''
demonstrates the use of the tf.data.Dataset API to build piplines to feed data to your model.It covers:
Creating a Dataset
Iteration over a Dataset with eager execution enabled

create a source Dataset
create a source datset using one of the factory functions like
Dataset.from_tensors,Dataset.from_tensor_slices or using objects that read from files like
TextLineDataset or TFRecordDataset.



'''

