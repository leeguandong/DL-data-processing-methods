# 用keras数据，这个数据集成了x，y值

from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
from matplotlib import offsetbox
from time import time

palette = np.array(sns.color_palette("hls", 21))

t0 = time()
X = []

for filename in ["gap_InceptionV3.h5"]:
    with h5py.File(filename, 'r') as h:
        X.append(np.array(h['train']))
        y = np.array(h['train_label'])

print('-------', len(X))
print('-------', len(X))

X = np.concatenate(X, axis=1)
print(len(X))
print(len(X[0]))
print(X)

style_label_file = 'style_names'
# style_label_file = style_label_file.encode('utf-8')

style_labels = list(np.loadtxt(style_label_file, str, delimiter='\n'))
print(style_labels)

for i in range(len(style_labels)):
    style_labels[i] = style_labels[i][1:]
print(style_labels)

X_tsne = TSNE(n_components=2, learning_rate=100, random_state=0).fit_transform(X)
print(X_tsne)

f = plt.figure(figsize=(16, 8))
ax = plt.subplot(aspect='equal')

for i in range(21):
    ax.scatter(X_tsne[y == i, 0], X_tsne[y == i, 1], c=palette[i], label=style_labels[i])
plt.legend(loc=2, numpoints=1, ncol=2, fontsize=12, bbox_to_anchor=(1.05, 0.8))
ax.axis('off')
# plt.savefig('t_sne.eps')
# plt.show()


# xs = X_tsne[:, 0]
# ys = X_tsne[:, 1]
# print(xs, ys)
#
# ax = sns.stripplot(x=xs, y=ys)

# zs = X_tsne[:, 2]

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(xs, ys, zs, c=style_labels, marker='o')

title = 't-SNE embedding of the feature vector (time %.2fs)' % (time() - t0)
plt.title(title)
plt.savefig('tsne1.png', dpi=150)
plt.show()

# import itertools
#
# print('end')
#
# import matplotlib
#
# markers = matplotlib.markers.MarkerStyle.filled_markers
#
# markers = marker = itertools.cycle(markers)
#
# f = plt.figure(figsize=(16, 8))
#
# ax = plt.subplot(aspect='equal')
#
# for i in range(21):
#     ax.scatter(X_tsne[y == i, 0], X_tsne[y == i, 1], marker=markers.next(), c=palette[i], label=style_labels[i])
# plt.legend(loc=2, numpoints=1, ncol=2, fontsize=12, bbox_to_anchor=(1.05, 0.8))
# ax.axis('off')
# plt.savefig('t_sne.eps')
# plt.show()
