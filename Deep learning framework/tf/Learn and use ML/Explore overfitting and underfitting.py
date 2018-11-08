# To develop better generalizing of the model,some techniques such as regularization.
'''
If a network can only afford to memorize a small number of patterns,the optimization process will force
it to focus on the most prominent patterns
'''



# so we think multi-hot encoding will be overfitting than embedding
'''
list([1, 14, 22, 16, 43, 530, 973, 1622, 1385, 65, 458, 4468, 66, 3941, 4, 173, 36, 256, 5,
25, 100, 43, 838, 112, 50, 670, 2, 9, 35, 480, 284, 5, 150, 4, 172, 112, 167, 2, 336, 385, 39,
4, 172, 4536, 1111, 17, 546, 38, 13, 447, 4, 192, 50, 16, 6, 147, 2025, 19, 14, 22, 4, 1920,
4613, 469, 4, 22, 71, 87, 12, 16, 43, 530, 38, 76, 15, 13, 1247, 4, 22, 17, 515, 17, 12, 16,
626, 18, 2, 5, 62, 386, 12, 8, 316, 8, 106, 5, 4, 2223, 5244, 16, 480, 66, 3785, 33, 4, 130,
12, 16, 38, 619, 5, 25, 124, 51, 36, 135, 48, 25, 1415, 33, 6, 22, 12, 215, 28, 77, 52, 5,
14, 407, 16, 82, 2, 8, 4, 107, 117, 5952, 15, 256, 4, 2, 7, 3766, 5, 723, 36, 71, 43, 530,
476, 26, 400, 317, 46, 7, 4, 2, 1029, 13, 104, 88, 4, 381, 15, 297, 98, 32, 2071, 56, 26,
141, 6, 194, 7486, 18, 4, 226, 22, 21, 134, 476, 26, 480, 5, 144, 30, 5535, 18, 51, 36, 28,
224, 92, 25, 104, 4, 226, 65, 16, 38, 1334, 88, 12, 16, 283, 5, 16, 4472, 113, 103, 32, 15, 16,
5345, 19, 178, 32])

word_indices: we will produce n 1000-demension 0 vector,and we get number location in sequence,and
replace 1 in location use for to add it.

return multi-hot encoding
'''

# Demonstrate overfitting
'''
The simply way to prevent overfitting is to reduce the size of the model.
number of learnable parameters in the model(which is determined by the number of layers and the number
of the units per layer).In deep learning,the number of learnable parameters in a model is often referred
to as the model's capacity.Intuitively,a model with more parameters will have more 'memorization capacity'
and therefore will be able to easily learn a perfect dictionary-like mapping between training samples
and their targets, a mapping without any generalization power.but this would be useless when making predictions
on unseen data.

On the other hand,if the network has limited memorization resources,it will not be able to learn the mapping
as easily. To minimize its loss,it will have to learn compressed representations that have more predictive
power.At the same time,if you make your model too small,it will have difficulty fitting to the training data.
There is a blance between 'to much capacity' and 'hot enough capacity'.

To find an appropriate model size,it's best to start with relatively few layers and parameters,then begin increasing
the size of the layers or assing new layers until you see diminishing return on the validation loss
'''

# create a baseline model
'''
keras.model

model.compile

model.fit
'''

# Create a bigger model
'''
As an exercise,you can create an even larger model,and see how quickly it begins overfitting.Next,let's add to this
benchmark a network that has much more capacity, far more than the problem would warrant.
16
16
1
'''

# Create a smaller model
'''
4
4
1
'''

# Create a bigger model
'''
512
512
1
'''

# Plot the training and validation loss
'''
The soild lines show the training loss,and the dashed lines shows the validation loss,and we know a lower validation loss
indicates a better model.Here,the smaller network begins overfitting later than the baseline model(after 6 epochs rather
than 4) and its performance degrades much more slowly once it starts overfiting.

Notice that the larger network begins overfitting almost right away,after just one epoch,and overfit much more severely.
The more capacity the network has,the quickly it will be able to model the training dada(resulting in a low training loss)
but the more susceptible(敏感) it is tooverfitting.
'''

# Strategies
'''
Add weight regularization
This also applies to the models learned by neural network:given some training data and a network architecture, there are
multiple sets of weights values(multiple models) that could explain data,and simpler models are less likely to overfit
than complex ones.

A simple model in this context is a model where the distribution of parameter values has less entropy(or a model with fewer
parameters altogether,as we saw in the section above).Thus a common way to mitigate overfitting is to put constrains on
the complexity of a network by forcing its weights only to take small values,which makes the distribution of weights values
more 'regular'.This is called 'weight regularization'. and it is done by adding to the loss function of the network a cost
associated with having large weights.This comes in two flavors:
1. L1 regularization:
2. L2
Don't let the different name confuse you:weight decay is mathematically the exact same as L2 regularization

l2(0.001) means that every coefficient in the weight matrix of the layer will add 0.001*weight_coefficeient_value**2 to
the total loss of the network.Note that because this penalty is only added at training time,the loss for this network
will be much higher at training than at test time.

Add dropout
At test time,no units are dropped out,and instead the layer's output values are scaled down by a factor equal to the dropout
rate,so as to balance for the fact that more units are active than at training time.

To recap: here the most common ways to prevent overfitting in neural network
Get more training data
Reduce the capacity of the network
Add weight regularization
Add dropout
'''

