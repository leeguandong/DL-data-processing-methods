# 显示特征与特征之间的相关性
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)
import seaborn as sns

sns.set()

uniform_data = np.random.rand(3, 3)
print(uniform_data)
beatmap = sns.heatmap(uniform_data, vmin=0, vmax=1)

# 有正有负的数据，以0为中心的
# uniform_data = np.random.randn(3, 3)
# print(uniform_data)
# beatmap = sns.heatmap(uniform_data, center=0)

flights = sns.load_dataset('flights')
flights.head()

# 把上述数据转换成横轴为month，纵轴是year，坐标点为passengers
flights = flights.pivot('month', 'year', 'passengers')
print(flights)
# annot是将数字显示到图上，fmt是显示数字的格式,linewidths指定格与格之间的间距，cmap是调色板,cbar是条例
ax = sns.heatmap(flights, annot=True, fmt='d', linewidths=.5)
plt.show()
plt.savefig('热图',dpi=150)
