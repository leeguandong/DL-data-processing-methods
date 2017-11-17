# 柱状图
import pandas as pd
reviews = pd.read_csv('fandango_scores.csv')
cols = ['FILM','RT_user_norm']
norm_reviews = reviews[cols]

import matplotlib.pyplot as plt
from numpy import arange
num_cols = ['RT_user_norm','Metacritic_user_norm']

# ix索引
bar_heights = norm_reviews.ix[0,num_cols].values
bar_positions = arange(5)+0.75
fig,ax=plt.subplots()
ax.bar(bar_positions,bar_heights,0.3) # 第三个变量是柱的宽度
# ax.barh()
plt.show()

# # 散点图
# fig,ax=plt.subplots()
# ax.scatter(norm_reviews[''],norm_reviews[''])
# ax.set_xlabel()
# ax.set_ylabel()
# plt.show()

# boxplot


