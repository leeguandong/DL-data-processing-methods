# seaborn是对matplotlib的一种封装
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib as mpl

def sinplot(flip=1):
    x = np.linspace(0, 14, 100)
    for i in range(1, 7):
        plt.plot(x, np.sin(x + i * 0.5) * (7 - i) * flip)

sns.set()
sinplot()

# 5种主题风格
# darkgird  whitegrid  dark  white  ticks
sns.set_style('whitegrid')
data = np.random.normal(size=(20, 6)) + np.arange(6) / 2
sns.boxplot(data=data)

# offset 图距离轴线距离

# 在with里面执行的一个风格，在with外面执行的一个风格
with sns.axes_style('darkgrid'):
    plt.subplot()
    sinplot()
plt.subplot(212)
sinplot(-1)

# sns.set_context('') 有各种风格，比如paper、poster、notebook

plt.show()