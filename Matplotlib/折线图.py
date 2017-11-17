import pandas as pd

unrate = pd.read_csv('unrate.csv')
unrate['DATE'] = pd.to_datetime(unrate['DATE'])

import matplotlib.pyplot as plt

plt.plot()
plt.show()

first_twelve = unrate[0:12]  # 先取前12个数据
plt.plot(first_twelve['DATE'], first_twelve['VALLE'])
plt.xticks(rotation=45)
plt.xlabel('Month')
plt.ylabel('Rate')
plt.title('Unemployment Trends')
plt.show()
