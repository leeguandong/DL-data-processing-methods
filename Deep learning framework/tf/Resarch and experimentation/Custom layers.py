'''
We recommend using tf.keras as a high-level API for building neural network.
That said,most Tensorflow APIs are usable with eager execution.
'''

# Layers:common sets of useful operations

# Implementing custom layers(自定义层)
'''
The best way to implement your own layer is extending the tf.keras.Layer class and implementing:
__init__ ,所有输入相关的初始化。 __build__,where you know the shape s of the input tensors and can do
the rest of the initalization,__call__ where you do the forward computation
'''

# Models:composing layers
'''
Many interesting layer-like things in machine learning models are implemented by composing exist layers
'''

