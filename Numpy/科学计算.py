import numpy as np

a = np.array([10, 20, 30, 40])
b = np.arange(4)

A = np.array([
    [1, 2],
    [3, 4]
])

B = np.array([
    [1, 2],
    [3, 4]
])

print(A * B)  # 对应位置相乘
print(A.dot(B))  # 矩阵乘法

# 矩阵操作
# np.floor 向下取整
m = np.floor(10 * np.random.random((3, 4)))
print(m)
# 将m整平的操作，很重要的一个操作
print(m.ravel())
m.shape = (6, 2)
# 12个元素，分3行，那么一定是4列，没必要写上去了，写-1相当于一个默认值，numpy自己计算
m.reshape(3, -1)

# 矩阵之间的组合和分解
k = np.floor(10 * np.random.random((2, 2)))
j = np.floor(10 * np.random.random((2, 2)))
print(k)
print(j)
# 横着拼
print(np.hstack((k, j)))
# 竖着拼
print(np.vstack((k, j)))

k = np.floor(10 * np.random.random((2, 12)))
# 横着切
print(np.hsplit(k, 3))
# 指定位置切
print(np.hsplit(k, (3, 4)))

# 复制问题
# view() 浅拷贝 虽然复制的id不同，但是共用同一值
# copy() 深拷贝

# 排序
data = np.sin(np.arange((20)).reshape(5, 4))
print(data)
# argmax返回的是最大值所在行序号，不是值，是序号
index = data.argmax(axis=0)
print(index)
# 第一个参数是行，第二个参数是列
data_max = data[index, range(data.shape[1])]
print(data_max)

# 矩阵扩张
a = np.arange(0, 40, 10)
b = np.tile(a, (2, 2))
print(b)

# 排序
a = np.array([4, 3.1, 2])
j = np.argsort(a)
print(a[j])
