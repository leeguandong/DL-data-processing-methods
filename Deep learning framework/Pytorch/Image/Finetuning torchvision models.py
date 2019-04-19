'''
two types of transfer learning finetuning and feature extraction.

In finetuning,we start with a pretrained model and updated all of the model's parameters for our network.
In feature extraction,we start with a pretrained model and only updated the final layer weights from
which we drive prediction.It is called feature extraction because we use the pretrained CNN as a
feature-extractor,and only change the output layer.
'''

# In general both transfer learning methods foolow the same few steps:
'''
Initialize the pretrained model
Reshape the final layer(s) to have the same number of outputs as the number of classes in the new dataset
Define for the optimization algorithm which parameters we want to update during training
Run the training step
'''

# Inputs

# 模型训练和验证代码
'''
train_model function handles the training and validation of a given model.
a dictionary of dataloader,a loss function,an optimizer,a specified number of epochs to train and validate for,
and a boolean flag for when the model is an Inception model.
'''