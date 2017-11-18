import numpy as np
from numpy import linalg
from numpy.linalg import norm
from scipy.spatial.distance import squareform, pdist

import sklearn
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale
from sklearn.metrics.pairwise import paired_distances
from sklearn.manifold.t_sne import _joint_probabilities, _kl_divergence
from sklearn.utils.extmath import _ravel

# 随机状态值
RS = 20171111

# 图形库matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib

# 使用seaborn更好的绘图
import seaborn as sns

sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context('notebook', font_scale=1.5, rc={'lines.linewidth': 2.5})

# 使用matplot和moviepy生成动画
# from moviepy.video.io.bindings import mplfig_to_npimage
# import moviepy.editor as mpy

# 加载手写数字识别库，共有1797张图片，每张大小8*8
digits = load_digits(n_class=6)
print(digits.data.shape)
print(digits['DESCR'])

nrows, ncols = 2, 5
plt.figure(figsize=(6, 3))  # 图片尺寸
plt.gray()
for i in range(ncols * nrows):
    ax = plt.subplot(nrows, ncols, i + 1)
    ax.matshow(digits.images[i, ...])
    plt.xticks([])
    plt.yticks([])
    plt.title(digits.target[i])
plt.savefig('digits-generated.png', dpi=150)

# 运行t-SNE算法
# X拿图，y拿标签
X = np.vstack([digits.data[digits.target == i] for i in range(10)])
# print(X)
y = np.hstack([digits.target[digits == i] for i in range(10)])
# print(y)
digits_proj = TSNE(random_state=RS).fit_transform(X)


# 显示转换后数据集的函数
def scatter(x, colors):
    # 我们使用seaborn中的调色板
    palette = np.array(sns.color_palette('hls', 10))

    # 我们创建一个散点图
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:, 0], x[:, 1], lw=0, s=40, c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # 我们为每个数字添加标签
    txts = []
    for i in range(10):
        # 每一个标签的位置
        # 返回两个列的平均值
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        # 添加文字信息，把两个列平均值添加上去
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground='w'),
            PathEffects.Normal()
        ])
        txts.append(txt)

    return f, ax, sc, txts

scatter(digits_proj, y)
plt.savefig('image.png', dpi=120)
