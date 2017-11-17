import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

np.random.seed(sum(map(ord, 'axis_grids')))
tips = sns.load_dataset('tips')
tips.head()

# 将FacetGrid实例化出来
g = sns.FacetGrid(tips, col='time')
g.map(plt.hist, 'tip')

# 散点图把数据描绘出来 alpha透明程度，越小越透明
g = sns.FacetGrid(tips, col='sex', hue='smoker')
g.map(plt.scatter, 'total_bill', 'tip', alpha=.1)
g.add_legend()
plt.show()

# 在FacetGrid中指定要花图的类型即可

from pandas import Categorical

ordered_days = tips.day.value_counts().index
print(ordered_days)

g = sns.FacetGrid(tips, row='day', row_order=ordered_days, size=1.7, aspect=4)
g.map(sns.boxplot, 'total_bill')

# FacetGrid绘制多变量
pal = dict(Lunch='seagreen', Dinner='gray')
g = sns.FacetGrid(tips, hue='time', palette=pal)
g.map(plt.scatter, 'total_bill', 'tip', s=50, alpha=.7, linewidth=.5, edgecolor='white')
g.add_legend() # 标签项

g.set_axis_labels('Total bill','Tip') # x轴和y轴的名字
g.set(xticks=[10,30,50],yticks=[2,6,10]) # 给x、y轴设置范围
g.fig.subplots_adjust(wspace=.02,hspace=0.02) # 设置子图与子图之间的间隔

############
iris = sns.load_dataset('iris')
g=sns.PairGrid(iris)  # 可以指定对角线上画什么，非对角线画什么
g.map(plt.scatter)
plt.show()














