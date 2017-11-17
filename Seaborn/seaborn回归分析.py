import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

sns.set(color_codes=True)
np.random.seed(sum(map(ord, 'regression')))

tips = sns.load_dataset('tips')
tips.head()

# regplot()和mplot()都可以绘制回归关系
sns.regplot(x='total_bill', y='tip', data=tips)

# x_jitter=.05 抖动值

#############
# 多因素
# 盒图  大于四分之三+N或者小于四分之一-N为离群点
# 离群点

# swarmplot
# violinpolt
# boxplot

# 显示值的集中趋势可以用条形图
# sns.barplot(x='sex',y='survived',hue='class',data=titanic)
# hue属性就是在图上对比的因素

# 点图可以更好的描述变化差异
# sns.pointplot(x='sex',y='survived',hue='class',data=titanic)
# sns.pointplot(x='class',y='survived',hue='sex',data=titanic,palette={'male':'g','female':'m'},
#               markers=['^','o'],linestyles=['-','--'])
# palette调色板，指定颜色，markers指定标记，linestyles指定线型

# 多层面板分类图
# factorplot只要调整kind，就可以显示各种类型图
sns.factorplot(x='day',y='total_bill',hue='smoker',data=tips,kind='bar')







