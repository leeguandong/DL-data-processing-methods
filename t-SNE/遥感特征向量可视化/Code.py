from numpy import genfromtxt
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

fn = r'electricity_price_and_demand_20170926.csv'
my_data = genfromtxt(fn, delimiter=',')

model = KMeans(n_clusters=5)
model.fit(my_data)
labels = model.predict(my_data)
print(labels)

model = TSNE(n_components=3, learning_rate=100, random_state=0)
transformed = model.fit_transform(my_data)
print(transformed)

xs = transformed[:, 0]
ys = transformed[:, 1]
zs = transformed[:, 2]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xs, ys, zs, c=labels, marker='o')
plt.show()
