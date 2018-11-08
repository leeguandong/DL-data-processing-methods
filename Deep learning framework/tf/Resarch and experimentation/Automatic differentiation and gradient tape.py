'''
A key technique for optimizing machine learning models
'''

# Derivatives of a function
'''
Tensorflow provides APIs for automatic differentiation - computing the derivative of a function. The way
that more closely mimics the math is to encapsulate in a python fucntion.say f,and use tfe.gradients_function
to create a function that computes the derivations of f with respect to its arguments.
'''

# 高阶梯度

# Gradient taps
'''
Every differentiable Tensorflow operation has an associated gradient function.For example,the gradient
function of tf.square(x) would be a function that returns 2.0*x.To compute the gradient of a user-defined
function(like f(x) in the example above),Tensorflow first 'records' all the operations applied to
compute the output of the function.We call this record a 'tape'.

有的时候把感兴趣的计算封装到一个函数是比较麻烦的，比如我想要计算函数中每一个输出梯度的中间值，在这种情况下，tf.GradientTape
中内存是有用的。所有的计算在tf.GrandientTape中都是记录下来的。

Higher-order gradient
高阶导数
'''



