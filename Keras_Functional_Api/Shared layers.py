import keras
from keras.layers import Input, LSTM, Dense
from keras.models import Model

'''
首先我们将一条推特转换为一个尺寸为 (140, 256) 的矩阵，即每条推特 140 字符，每个字符为 256 维的 one-hot 编码 （取 256 个常用字符）。
'''
tweet_a = Input(shape=(140, 256))
tweet_b = Input(shape=(140, 256))

'''
要在不同的输入上共享同一个层，只需实例化该层一次，然后根据需要传入你想要的输入即可
'''
# 这一层可以输入一个矩阵，并返回一个 64 维的向量
shared_lstm = LSTM(64)

# 当我们重用相同的图层实例多次，图层的权重也会被重用 (它其实就是同一层)
encoded_a = shared_lstm(tweet_a)
encoded_b = shared_lstm(tweet_b)

# 然后再连接两个向量
merged_vector = keras.layers.concatenate([encoded_a, encoded_b], axis=-1)

# 再在上面添加一个逻辑回归层
predictions = Dense(1, activation='sigmoid')(merged_vector)

# 定义一个连接推特输入和预测的可训练的模型
model = Model(inputs=[tweet_a, tweet_b], outputs=predictions)
model.summary()
data_a = []
data_b = []
labels = []

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit([data_a, data_b], labels, epochs=10)

'''
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to
==================================================================================================
input_1 (InputLayer)            (None, 140, 256)     0
__________________________________________________________________________________________________
input_2 (InputLayer)            (None, 140, 256)     0
__________________________________________________________________________________________________
lstm_1 (LSTM)                   (None, 64)           82176       input_1[0][0]
                                                                 input_2[0][0]
__________________________________________________________________________________________________
concatenate_1 (Concatenate)     (None, 128)          0           lstm_1[0][0]
                                                                 lstm_1[1][0]
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 1)            129         concatenate_1[0][0]
==================================================================================================
Total params: 82,305
Trainable params: 82,305
Non-trainable params: 0
__________________________________________________________________________________________________
'''
