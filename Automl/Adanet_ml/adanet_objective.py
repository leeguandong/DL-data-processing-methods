# The AdaNet objective
'''
One of key contributions from AdaNet:Adaptive structral learning of neural networks is defining an algorithm that aims to
directly minimize the deepboost generalization bound from deep boosting when applied to neural networks.
'''

# How adanet uses the objective
'''
1.To learn to scale/transform the outputs of each subnetwork h as part of the ensemble
2.To select the best candidate subnetwork h at each Adanet_ml iteration to include in the ensemble

Adanet_ml 惩罚更复杂的子网络，其混合权重具有更大的L1正则化，并且在每次迭代时不太可能选择更复杂的子网添加整体。使用 Adanet_ml 学习集合的混合权重和
执行候选选择的好处。
'''

# Boston housing dataset

# Download the data

# Supply the data in tensorflow
'''
Our first task is supply the data in tensorflow. Using the tf.estimator.Estimator convention,we will define a function that
returns an input_fn which returns feature and label tensors.

we will also use the tf.data.Dataset Api to feed the data into our models

Also,as a preprossing step,we will apply tf.log1p to log-scale the features and labels for improved numerical stability
during training. To recover the model's predictions in the correct scale,you can apply tf.math.expml to the prediction.
'''

# Define the subnetwork generator
'''
Let's define a subnetwork generator similar to the one in adanet and in simple_dnn.py which creates two candidate fully-connected
neural network at each iteration with the same width,but one an additional hidden layer.To make our generator adaptive,each
subnetwork will have at least the same number of hidden layers as the most recently added subnetwor to the previous_ensemle.


      optimizer: An `Optimizer` instance for training both the subnetwork and the mixture weights.
      layer_size: The number of nodes to output at each hidden layer.
      num_layers: The number of hidden layers.
      learn_mixture_weights: Whether to solve a learning problem to find the
        best mixture weights, or use their default value according to the
        mixture weight type. When `False`, the subnetworks will return a no_op
        for the mixture weight train op.
      seed: A random seed.

    Returns:
      An instance of `_SimpleDNNBuilder`.

Generates a two DNN subnetworks at each iteration.

  The first DNN has an identical shape to the most recently added subnetwork
  in `previous_ensemble`. The second has the same shape plus one more dense
  layer on top. This is similar to the adaptive network presented in Figure 2 of
  [Cortes et al. ICML 2017](https://arxiv.org/abs/1607.01097), without the
  connections to hidden layers of networks from previous iterations.
'''

# Train and evaluate
'''
Next we create an adanet.Estimator using the SimpleDNNGenerator we just defined

In this section we will show the effects of two hyperparamters:learning mixture weights and complexity regularization

On the righthand side you will be able to play with the hyperparameters of this model.Until you reach the end of this section,
we ask that you not change them

At first we will not learn the mixture weights,using their default initial value.

这些超参数推出了一个在测试集上达到0.0348 MSE的模型。 请注意，整体由5个子网组成，每个子网都比前一个更深。
最复杂的子网由5个隐藏层组成。

由于SimpleDNNGenerator产生不同复杂度的子网，并且我们的模型给予每个子网相同的权重，因此AdaNet选择了在每次迭代时
最大程度降低集合训练损失的子网，可能是具有最多隐藏层的子网，因为它具有最大容量，并且 我们还没有惩罚更复杂的子网。

接下来，让我们使用SGD将混合权重作为凸优化问题来学习，而不是为每个子网分配相等的权重
'''

# Conclusion
'''
在本教程中，您可以探索使用训练AdaNet模型的混合权重。您还可以通过总是在每次迭代中选择最佳候选子网来构建整体而构建整体，
这是基于它能够改善整体在训练集上的损失，并对其结果求平均值。

统一平均合奏在实践中工作得不合理，但是当候选人具有不同的复杂性时，使用$ \ lambda $和$ \ beta $的正确值来学习混合权重
应始终产生更好的模型。但是，这确实需要一些额外的超参数调整，因此实际上您可以首先使用默认混合权重和$ \ lambda = 0 $
训练AdaNet，并且一旦确认子网正确训练，就可以调整混合权重超参数。

虽然这个例子探讨了一个回归任务，但这些观察结果适用于在二元分类和多类分类等其他任务中使用AdaNet。
'''
