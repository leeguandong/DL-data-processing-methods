import numpy as np
import pandas as  pd

titanic_survival = pd.read_csv('train.csv')
print(titanic_survival.head())

age = titanic_survival['Age']
print(age.loc[0:10])

# 判断缺失值
age_is_null = pd.isnull(age)
print(age_is_null)

# 缺失值是true，缺失值都会保存下来
age_null_true = age[age_is_null]
print(age_null_true)

# 打印缺失值的多少
age_null_count = len(age_null_true)
print(age_null_count)

# 如果程序中有缺失值，求不出来平均值
mean_age = sum(titanic_survival['Age']) / len(titanic_survival['Age'])
print(mean_age)

# 正确处理
correct_mean_age = titanic_survival['Age'].mean()
print(correct_mean_age)

# 每个仓位等级的平均船票价
passenger_classes = [1, 2, 3]
fares_by_class = {}
for this_class in passenger_classes:
    pclass_rows = titanic_survival[titanic_survival['Pclass'] == this_class]
    pclass_fares = pclass_rows['Fare']
    fare_for_class = pclass_fares.mean()
    fares_by_class[this_class] = fare_for_class
print(fares_by_class)

# 计算每个仓位获救的平均值
passenger_survival = titanic_survival.pivot_table(index='Pclass', values='Survived', aggfunc=np.mean)
print(passenger_survival)

# 每个仓位等级的年龄平均值,如果制定agefunc，则按照指定函数去计算，如不指定，则按照求均值计算
passenger_classes = titanic_survival.pivot_table(index='Pclass', values='Age')
print(passenger_classes)

# 求每个码头的票价总数和获救人口两个维度的影响
passenger_classes = titanic_survival.pivot_table(index='Embarked', values=['Fare', 'Survived'], aggfunc=np.sum)
print(passenger_classes)

# 去掉缺失值
new_titanic_survival = titanic_survival.dropna(axis=0, subset=['Age', 'Sex'])
print(new_titanic_survival)

# 排序
new_titanic_survival = titanic_survival.sort_values('Age', ascending=False)
print(new_titanic_survival)

# 一个从0开始的新索引
new_titanic_survival = titanic_survival.reset_index(drop=True)
print(new_titanic_survival)

# 自定义函数操作
# 返回第100行数据
def hundredth_row(column):
    hundredth_item = column.loc[99]
    return hundredth_item

hundredth_row = titanic_survival.apply(hundredth_row)
print(hundredth_row)

# 返回缺失值个数
def not_null_count(column):
    # 返回True值，是缺失值
    column_null = pd.isnull(column)
    # 把true放到列表
    null = column[column_null]
    # 计算缺失值个数
    return len(null)

column_null_count = titanic_survival.apply(not_null_count)
print(column_null_count)

