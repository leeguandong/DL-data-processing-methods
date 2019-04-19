# 循环神经网络基类
import keras
from keras import backend as K
from keras.models import Model
from keras.utils import plot_model

# 首先，让我们定义一个 RNN ，作为网络子类。
class MinimalRNNCell(keras.layers.Layer):
    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = units
        super(MinimalRNNCell, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                      initializer='uniform', name='kernel')
        self.recurrent_kernel = self.add_weight(shape=(self.units, self.units),
                                                initializer='uniform', name='recurrent_kernel')
        self.built = True

    def call(self, inputs, states):
        prev_output = states[0]
        h = K.dot(inputs, self.kernel)
        output = h + K.dot(prev_output, self.recurrent_kernel)
        return output, [output]

# 让我们在 RNN 层使用这个单元
# cell = MinimalRNNCell(32)
# x = keras.Input((None, 5))
# layer = keras.layers.RNN(cell)
# y = layer(x)

# 构建堆叠的 RNN 的方法, Input( timesteps,input_dim ),timesteps 是时间维度
cells = [MinimalRNNCell(32), MinimalRNNCell(64)]
x = keras.Input((100, 5))
layer = keras.layers.RNN(cells)
y = layer(x)

model = Model(inputs=x, outputs=y)
model.compile(loss="categorical_crossentropy", optimizer="sgd")
model.summary()
plot_model(model=model, to_file='model.png', show_shapes=True)
