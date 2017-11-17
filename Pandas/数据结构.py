import pandas

# 数据读取
food_info = pandas.read_csv(" ")

# 显示数据，但是只显示前5条
food_info.head()
# 显示数据的后几行
food_info.tail()
print(food_info.columns)
print(food_info.shape)

# 索引与计算
# 取第一个数据,到那时按行取
print(food_info.loc[0])
# food_info.loc[3:6]

# 寻找以g为结尾的列名,很简单的python操作
col_names = food_info.columns.tolist()
print(col_names)
gram_columns = []

for c in col_names:
    if c.endswith('g'):
        gram_columns.append(c)
gram_df = food_info[gram_columns]
print(gram_df.head(3))

# * 对应位置相乘
# 归一化
# 先找到归一化的列，然后除以这个列中的最大值

# 排序
food_info.sort_values('', inplace=True, ascending=False)

# Series结构
import pandas as pd

fandango = pd.read_csv('')
series_film = fandango['FILM']
series_rt = fandango['RottenTomatoes']

from pandas import Series

film_names = series_film.values
rt_scores = series_rt.values

series_custom = Series(rt_scores, index=film_names)

# 排序，很少有人用Series进行排序
original_index = series_custom.index.tolist()
sorted_index = sorted(original_index)
sorted_by_index = series_custom.reindex(sorted_index)
print(sorted_by_index)


