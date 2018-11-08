'''
Tensrflow includes a higher-level neural network API which prodeives useful abstractions to reduce boilerplate
'''

# Variables
'''
Computations using Variables are automatically traced when computing gradient.
'''

# Example:Fitting a linear model
'''
Tensor,GradientTape,Variable

4 setps:
1. Define the model
2. Define a loss function
3. Obtain training data
4. Run throught the training data and use an 'optimizer' to adjust the variables to fit the data

基础模型是一个线性模型，wx+b，每一轮就是不断的求导去逼近真实的w，b，标签是outputs，加了噪音，最终的w，b通过随机初始化的数据也是逼近或者说模拟的
函数的，通过不断求导这种方式，实际上就是反向传播了。
'''

'''
In theory, this is pretty much all you need to use TensorFlow for your machine learning research.
In practice, particularly for neural networks, the higher level APIs like tf.keras will be much more
convenient since it provides higher level building blocks (called "layers"), utilities to save and restore state,
a suite of loss functions, a suite of optimization strategies etc.


'''
