import matplotlib.pyplot as plt
import pandas as pd

# fig = plt.figure()  # 定义画图区域
# ax1 = fig.add_subplot(2, 2, 1)
# ax2 = fig.add_subplot(2, 2, 2)
# ax3 = fig.add_subplot(2, 2, 4)
# plt.show()

import numpy as np

# figsize控制子图大小
fig = plt.figure(figsize=(3, 3))
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)

ax1.plot(np.random.randint(1, 5, 5), np.arange(5))
plt.show()

unrate = pd.read_csv('unrate.csv')
fig = plt.figure(figsize=(10, 6))
colors = ['red', 'blue', 'green', 'orange', 'black']
for i in range(5):
    start_index = i * 12
    end_index = (i + 1) * 12
    subset = unrate[start_index:end_index]
    label = str(1948 + i)
    plt.plot(subset['MONTH'], subset['VALUE'], c=colors[i], label=label)
plt.legend(loc='best')
plt.show()
