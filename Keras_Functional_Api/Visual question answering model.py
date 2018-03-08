from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers import Input, LSTM, Embedding, Dense
from keras.models import Model, Sequential
import keras

'''
当被问及关于图片的自然语言问题时，该模型可以选择正确的单词作答。
它通过将问题和图像编码成向量，然后连接两者，在上面训练一个逻辑回归，来从词汇表中挑选一个可能的单词作答。
'''

# 首先，让我们用 Sequential 来定义一个视觉模型。这个模型会把一张图像编码为向量。
vision_model = Sequential()
vision_model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(224, 224, 3)))
vision_model.add(Conv2D(64, (3, 3), activation='relu'))
vision_model.add(MaxPooling2D((2, 2)))
vision_model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
vision_model.add(Conv2D(128, (3, 3), activation='relu'))
vision_model.add(MaxPooling2D((2, 2)))
vision_model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
vision_model.add(Conv2D(256, (3, 3), activation='relu'))
vision_model.add(Conv2D(256, (3, 3), activation='relu'))
vision_model.add(MaxPooling2D((2, 2)))
vision_model.add(Flatten())

# 现在让我们用视觉模型来得到一个输出张量
image_input = Input(shape=(224, 224, 3))
encoded_image = vision_model(image_input)

# 接下来，定义一个语言模型来将向量来将问题编码成一个向量，每个向量最长100个词，词的索引从1到9999
question_input = Input(shape=(100,), dtype='int32')
embedded_question = Embedding(input_dim=10000, output_dim=256, input_length=100)
encoded_question = LSTM(265)(embedded_question)

# 连接向量和图像向量
merhed = keras.layers.concatenate([encoded_question, encoded_image])

# 然后在上面训练一个1000词的逻辑回归模型
output = Dense(1000, activation='softmax')(merhed)

# 最终模型
vqa_model = Model(inputs=[image_input, question_input], outputs=output)

# 下一步就是在真实数据上训练模型



'''
现在我们已经训练了图像问答模型，我们可以很快地将它转换为视频问答模型。在适当的训练下，你可以给它展示一小段视频（例如 100 帧的人体动作），然后问它一个关于这段视频的问题（例如，「这个人在做什么运动？」 -> 「足球」）。
'''
from keras.layers import TimeDistributed

video_input = Input(shape=(100, 224, 224, 3))

#### 这是基于之前定义的视觉模型（权重被重用）构建的视频编码
# 输出为向量的序列
encoded_frame_sequence = TimeDistributed(vision_model)(video_input)
# 输出为一个向量
encoded_video = LSTM(256)(encoded_frame_sequence)

# 这是问题编码器的模型级表示，重复使用与之前相同的权重
question_encoder = Model(inputs=question_input, outputs=encoded_question)

# 让我们用它来编码这个问题：
video_question_input = Input(shape=(100,), dtype='int32')
encoded_video_question = question_encoder(video_question_input)

# 视频问答模式
merged = keras.layers.concatenate([encoded_video, encoded_video_question])
output = Dense(1000, activation='softmax')(merhed)
video_qa_model = Model(inputs=[video_input, video_question_input], outputs=output)
