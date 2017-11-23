from time import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import (manifold, datasets, decomposition, ensemble, discriminant_analysis, random_projection)

# 加载数据集
digits = datasets.load_digits(n_class=6)
x = digits.data
y = digits.target
n_samples, n_feature = x.shape
n_neighbors = 30
# 1083,64
print(n_samples, n_feature)


# 可视化和调整嵌入向量的尺寸
# 这个代码可以复用的，就是归一化和可视化
def plot_embedding(X, title=None):
    # 归一化操作
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(digits.target[i]),
                 # plt.cm.Set1 颜色图
                 color=plt.cm.Set1(y[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    # 对于图像点展示
    if hasattr(offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(digits.data.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 4e-3:
                # don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [X[i]]]
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r),
                X[i])
            ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)


# 描绘输入图像
# 此处是一个20*20的输入图像  input images.png
n_img_per_row = 20
img = np.zeros((10 * n_img_per_row, 10 * n_img_per_row))
for i in range(n_img_per_row):
    ix = 10 * i + 1
    for j in range(n_img_per_row):
        iy = 10 * j + 1
        img[ix:ix + 8, iy:iy + 8] = x[i * n_img_per_row + j].reshape((8, 8))

plt.imshow(img, cmap=plt.cm.binary)
plt.xticks([])
plt.yticks([])
plt.title('A selection from the 64-dimensional digits dataset')
plt.savefig('input images.png', dpi=150)

# # 随机二维投影使用随机酉矩阵
# # 随机投影
# print("Computing random projection")
# rp = random_projection.SparseRandomProjection(n_components=2, random_state=42)
# X_projected = rp.fit_transform(x)
# plot_embedding(X_projected, "Random Projection of the digits")
# plt.savefig('random projection.png', dpi=150)
#
# # 主成分分析法降维，只取前两个主成分
# print("Computing PCA projection")
# t0 = time()
# X_pca = decomposition.TruncatedSVD(n_components=2).fit_transform(x)
# plot_embedding(X_pca,
#                "Principal Components projection of the digits (time %.2fs)" %
#                (time() - t0))
# plt.savefig('PCA projection.png', dpi=150)
#
# # LDA 线性判别式分析降维，前2个线性判别成分
# # 寻找低维空间，这样的空间满足同类之间的距离最小，异类之间的距离最大
# print("Computing Linear Discriminant Analysis projection")
# X2 = x.copy()
# X2.flat[::x.shape[1] + 1] += 0.01  # Make X invertible
# t0 = time()
# X_lda = discriminant_analysis.LinearDiscriminantAnalysis(n_components=2).fit_transform(X2, y)
# plot_embedding(X_lda,
#                "Linear Discriminant projection of the digits (time %.2fs)" %
#                (time() - t0))
# plt.savefig('LDA projection.png', dpi=150)
#
# # 下面这些降维方法都是流形学习的方法
# # Isomap降维 通过点之间的距离矩阵进行降维
# print("Computing Isomap embedding")
# t0 = time()
# X_iso = manifold.Isomap(n_neighbors, n_components=2).fit_transform(x)
# print("Done.")
# plot_embedding(X_iso,
#                "Isomap projection of the digits (time %.2fs)" %
#                (time() - t0))
# plt.savefig('Isomap projection.png', dpi=150)
#
# # LLE 非线性降维方法
# # 保持在某种局部结构的条件下实现数据的整体降维，期间是没有显式的映射矩阵
# print("Computing LLE embedding")
# clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2,
#                                       method='standard')
# t0 = time()
# X_lle = clf.fit_transform(x)
# print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
# plot_embedding(X_lle,
#                "Locally Linear Embedding of the digits (time %.2fs)" %
#                (time() - t0))
# plt.savefig('LLE projection.png', dpi=150)
#
# # 修改之后的lle
# print("Computing modified LLE embedding")
# clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2,
#                                       method='modified')
# t0 = time()
# X_mlle = clf.fit_transform(x)
# print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
# plot_embedding(X_mlle,
#                "Modified Locally Linear Embedding of the digits (time %.2fs)" %
#                (time() - t0))
# plt.savefig('modified LLE projection.png', dpi=150)
#
# # HLLE embedding of the digits dataset
# print("Computing Hessian LLE embedding")
# clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2,
#                                       method='hessian')
# t0 = time()
# X_hlle = clf.fit_transform(x)
# print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
# plot_embedding(X_hlle,
#                "Hessian Locally Linear Embedding of the digits (time %.2fs)" %
#                (time() - t0))
# plt.savefig('hlle.png', dpi=150)
#
# # LTSA embedding of the digits dataset
# print("Computing LTSA embedding")
# clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2,
#                                       method='ltsa')
# t0 = time()
# X_ltsa = clf.fit_transform(x)
# print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
# plot_embedding(X_ltsa,
#                "Local Tangent Space Alignment of the digits (time %.2fs)" %
#                (time() - t0))
# plt.savefig('ltsa.png', dpi=150)
#
# # MDS  embedding of the digits dataset
# print("Computing MDS embedding")
# clf = manifold.MDS(n_components=2, n_init=1, max_iter=100)
# t0 = time()
# X_mds = clf.fit_transform(x)
# print("Done. Stress: %f" % clf.stress_)
# plot_embedding(X_mds,
#                "MDS embedding of the digits (time %.2fs)" %
#                (time() - t0))
# plt.savefig('mds.png', dpi=150)
#
# # Random Trees embedding of the digits dataset
# # 随机树
# print("Computing Totally Random Trees embedding")
# hasher = ensemble.RandomTreesEmbedding(n_estimators=200, random_state=0,
#                                        max_depth=5)
# t0 = time()
# X_transformed = hasher.fit_transform(x)
# pca = decomposition.TruncatedSVD(n_components=2)
# X_reduced = pca.fit_transform(X_transformed)
#
# plot_embedding(X_reduced,
#                "Random forest embedding of the digits (time %.2fs)" %
#                (time() - t0))
# plt.savefig('rt.png', dpi=150)
#
# # Spectral embedding of the digits dataset
# print("Computing Spectral embedding")
# embedder = manifold.SpectralEmbedding(n_components=2, random_state=0,
#                                       eigen_solver="arpack")
# t0 = time()
# X_se = embedder.fit_transform(x)
#
# plot_embedding(X_se,
#                "Spectral embedding of the digits (time %.2fs)" %
#                (time() - t0))
# plt.savefig('spectral.png', dpi=150)

# t-SNE embedding of the digits dataset
print("Computing t-SNE embedding")
tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
t0 = time()
print(x.shape)
print(x)
X_tsne = tsne.fit_transform(x)

plot_embedding(X_tsne,
               "t-SNE embedding of the digits (time %.2fs)" %
               (time() - t0))
plt.savefig('tsne.png', dpi=150)

plt.show()
