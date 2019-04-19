# This trains a sequence to sequence model for Spanish to english translation using tf.keras and eager excution

# Download and prepare the dataset
'''
prepare the data
1. Add a start and end token to each sentence
2. Clean the scentences by removing special characters
3. Create a word index and reverse word index (dictionaries mapping from word → id and id → word).
4. Pad each sentence to a maximum length.
'''

# Limit the size of the dataset to experiment faster(optimal)
'''
训练全部大于100000句子的数据集需要花费很长时间，为了训练更快一点，我没限制数据集为30000个句子，当然翻译的质量会下降伴随着更少的数据
'''

# Create a tf.data dataset

# Write the encoder and decoder model
'''
we implement an encoder-decoder model with attention which you can read about in the tensorflow.
The input is put throught an encoder model which gives us the encoder output of shape (batch_size,max_length,hidden_size)
and the encoder hidden state of shape(batch_size,bidden_size)

NMT模型使用encoder读取源句子，然后编码得到一个“有意义”的 vector，即一串能代表源句子意思的数字。然后 decoder 将这个 vector 解码得到翻译，
就想图1展示的那样。这就是所谓的 encoder-decoder 结构。用这种方式，NMT 解决了在传统的 phrase-based 的翻译模型中所存在的局部翻译的问题
（local translation problem），它能够捕捉到语言中更长久的依赖关系，比如性别认同（gender agreements）、句法结构等，然后得到更流畅的翻译结果。

FC = Fully connected(dense) layer
EO = Encoder output
H = hidden state
X = input to the decoder

score = FC(tanh(FC(EO) + FC(H)))
attention weights = softmax(score,axis=1) softmax by defaut is applied on the last axis but here we want to apply it on the
1st axis,since the shape of score is (batxh_size,max_length,hidden_size).Max_length is the length of our input, since we are
trying to assign a weight to each input,softmax should be applied on the axis.
context vector = sum(attention weughts * EO,axis = 1) Same reason as above for chosing axis as 1
embedding out = The input to the decoder X is passed through an embedding layer
merged vector = concat(embedding output,content vector)
This merged vector is then given to the GRU
'''

# Define the optimizer and loss function

# Checkpoints(object-based saving)

# Training
'''
1.Pass the input through the encoder which return encoder output and the encoder hidden state
2.The encoder output,encoder hidden state and the decoder input(which is the start token) is passed to the decoder
3.The decoder returns the predictions and the decoder hidden state
4.The decoder hidden state is then passed back into the model and the predictions are used to calculate the loss
5.Use teacher forcing to decide the next input to the decoder
6.Teacher forcing is the technique where the target word is passed as the next input to the decoder
7.The final step is to calculate the gradients and apply it to the optimizer anf backpropagate
'''

# Translate
'''
The evaluate function is similar to the training loop,except we don't use teacher forcing here.
The input to the decoder at each time step is its previous predictions along with the hidden state and the encoder output.

stop predicting when the model predicts the end token
And store the attention weights for every time step
'''




