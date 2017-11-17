# 直方图
import seaborn as sns
import numpy as np

sns.set(color_codes=True)
np.random.seed(sum(map(ord, 'distributions')))

x = np.random.normal(size=100)
sns.distplot(x, kde=False)

import pandas as pd

# 根据均值和协方差生成数据
mean, cov = [0, 1], [(1, .5), (.5, 1)]
data = np.random.multivariate_normal(mean, cov, 200)
df = pd.DataFrame(data, columns=['x', 'y'])
print(df)

# 观测两个变量之间的分布关系最好用散点图
sns.jointplot(x='x', y='y', data=df)

iris = sns.load_dataset('iris')
# pairplot对角线上是单个变量的变化情况，费对角线上是两两变量之间的关系
sns.pairplot(iris)



