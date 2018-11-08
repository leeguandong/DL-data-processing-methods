'''
Use tf.estimator API in tensorflow to solve a benchmark binary classification problem.Estimators are tf's 最具有扩展性和生产力的模块。

根据年龄、教育、婚姻、财产来预测一个人的年收入
'''

# 使用 tf 中的模型库提供广泛而深入的模型，将根目录添加到Python路径

# Setup

# Download the official implementation
'''
要把 models 这个模块添加到 python path 中,我就不具体展开了，就把自己写一下就ok了
'''

# Read the US Census data
'''
This example uses the US Census Income Dataset from 1994 and 1995.We have provide the census_dataset.py
script to download the data and perform a little cleanup.

binary classification problem.

列其实就是特征维度，在特征维度中有两类值，一个连续不断的值，一个是有限的类别数目
'''

# Converting Data into Tensors
'''
input builder function
returns the following pair:
1.features   A dict from feature names to Tensors or SparseTensors containing batches of features.
2.labels     A Tensor containing batches of labels

this approach has severly-limited scalability.Larger dataset should be streamed from disk.The census_dataset.input_fn provides
an example of how to do this using tf.decode_csv and tf.data.TextLineDataset.
'''

# Selecting and Engineering Features for the Model
'''
定义了一个数字的特征和类别的特征，并且分别讨论了单一的这两个特征对最终分类结果的影响。

Selecting the crafting the right set of feature columns is key to learning a effective model. A feature column can be either
one of the raw inouts in the original features fict(a base feature column),or any new columns created using transformations
defined over one or multiple base columns(a derived feature columns)

Base Feature Columns
Numeric columns
最简单的还是数字特征，This indicates that a feature is a numeric value that should be input to the model directly.

The model will use the feature_column defnitions to the model input.You can inspect the resulting output using the input_layer
function.

The following will train and evaluate a model using only the age feature.

classifier = LinearClassifier()
classifier.train()
classifier.evaluate()

Similarly,we can define NumericColumn for each continus feature column that want to use in the model

You could retrain a model on these features by changing the feature_colums argument to the constructor.

非数字，是有类别的
Categorical columns
To define a feature column for a categorical feature,create a CategoricationColumn using one of the tf.feature_column.categorical_column * functions.

If you know the set of all possible feature values of a column-and there are only a few of them - use categorical_column_with_vocabulary_list.
Each key in the list is assigned an auto-incremented ID starting from 0.

This creates a sparse one-hot vector from the raw input feature.

The input_layer function we are using is designed for DNN models and expects dense input.
we must wrap it in a tf.feature_column.indicator_column to create

If we don't know the set of possible values in advamce,use the categorical_column_with_hash_bucket instead
http://www.skytech.cn/article/4865.html
tf 中表示非数值型特征，使用 tf.feature_column 这个函数，给特征建立（0,0，....1）这种形式

Here each possible value in the feature column  occupation  is hashed to an interger ID as we encounter them in training.The
example batch has a few different occupations.
feature column 变成一个整数ID

If we run input_layer with the hashed column,we see that the output shape is (batch_size,hash_bucket_size)

No matter how we choose to define a SparseColumn,each feature string is mapped into an interger ID by looking up a fixed mapping
or by hashing.Under the hood,the LinearModel class is responsible for manging the mapping and create tf.Variable to store the model
parameters(model weights) for each feature ID.

It's easy to use both sets of columns to configure a model that uses all these features.
'''

# Derived feature columns
'''
Making Continuous features categorical through Bucketzation
Sometimes the relationship between a continuous feature and the label is not linear.For example,age and income - a person's
income may grow in the early stage of their career,then the growth may slow at some point,and finally,the income decreases
after retirement.in this scenario,using the raw age as a real-valued feature column might not be a good choice because the
model can only learn one of the three cases.
1. Income always increase at some rate as age grows(positive correction)
2. Income always decrease at some rate as age grows(negative correction)
3. Income always stays the same no matter at what age(no correlation)

If we want to learn the fine-grained correction between and each age group separately,we can leverage bucketization.
bucketization is a process of dividing the entire range of a continuous feature into a set of consecutove buckets,and then
converting the original numerical feature into a bucket ID depending on which bucket that value falls into.

Learn complex relationships with crossed column
有些特征之间是相互关联的，就是两个特征维度的综合也是对最终的label产生重要影响的因子。

To learn the differences between differnt feature combinations,we can add crossed feature columns to the model.

We can also create a crossed_column over more than two columns.有些特征之间的交叉是非常严重的

使用 crossed columns 可以避免类别数目的快速增长。
'''

# Define the logistic regression model
'''
After processing the input data and defining all the data feature columns,we can put them together and build a logistic regression
model.The previous section showed several types of base and derived feature columns,including:
1. CategoricalColumn
2. NumericColumn
3. BucketizedCoolumn
4. CrossedColumn

All these are substract of the abstrcat FeatureColumn class and can be added to the feature_columns field of a model.
'''

# Train and evaluate the model
'''
Training a model is just a single command using the tf.estimator API

After the model is trained,evaluate the accuracy of the model by predicting the labels of the holdout data

After the model is evaluated,we can use it to predict whether an individual has an annual income of over 50000dollars given
an individual's information input.
In predict,we need not to shuffle the test data.
'''

# Adding Regularization to prevent overfiting
'''
Overfitting can occur when a model is excessively complex,such as having too many parameters relative to the number of
observed training data. Regularization allows you to control the model's complexity and make the model more generalizable
to unseen data.

you can use l1 and l2 regularization

但是正则化模型表现并不一定比基础模型要好。
两种类型的正则化都将权重的分布变成0。L2正则化对分布的尾部具有更大的影响，消除了极端的权重。L1正则化产生更精确的0值，在这种情况下，她将-200设置为0.
'''