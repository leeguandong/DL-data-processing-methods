import numpy as np

# world_alcohol = np.genfromtxt('',delimiter='，',dtype=str)

# ndarray是numpy中的核心结构

# 一维结构
vector = np.array([1, 2, 3, 4])
# 二维结构，就是numpy中的ndarray
matrix = np.array([[5, 10, 15], [20, 25, 35]])

# 输入numpy中的数据要保证是同一数据类型的，否则会存在自动转化
numbers = np.array([1, 2, 3, 4])
print(numbers.dtype)
print(numbers[0:3])

# 取矩阵中的一列
matrix = np.array([
    [5, 10, 15],
    [20, 25, 30],
    [35, 40, 45]
])
print(matrix[:, 1])
# 去两列
print(matrix[:, 0:2])

import numpy as np

vector = np.array([5, 10, 15, 20])
print(vector == 10)
# [False  True False False]

matrix = np.array([
    [5, 10, 15],
    [20, 25, 30],
    [35, 40, 45]
])
print(matrix == 10)

# 布尔值当索引
matrix = np.array([
    [5, 10, 15],
    [20, 25, 30],
    [35, 40, 45]
])
second_column_25 = (matrix[:, 1] == 25)
print(second_column_25)
print(matrix[second_column_25, :])
# 取出25所在的哪一行

# 或 与

vector = np.array(['1','2','3','4'])
print(vector.dtype)
vector=vector.astype(float)
print(vector.dtype)

# numpy.array求极值

# 维度 axis = 1 行维度  axis = 0 列维度

a = np.arange(15).reshape(3,5)
print(a)

print(a.shape)
print(a.ndim)  # 2维

# (3,4)必须加括号，因为(3,4)是一个元组
np.zeros((3,4))

# 3维
np.ones(2,3,4)

from numpy import pi
# 在0到2pi之间均匀产生100个点
np.linspace(0,2*pi,100)




