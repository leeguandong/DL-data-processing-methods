import numpy as np
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt

X = np.array([[0, 100, 0], [0,65, 1], [1, 99, 1], [25, 1, 1]])
X_embedded = TSNE(n_components=2).fit_transform(X)
print(X_embedded)
x = X_embedded[:, 0]
y = X_embedded[:, 1]
print(x, y)
ax = sns.stripplot(x=x, y=y)
plt.show()
plt.savefig('tsne散点图', dpi=150)
