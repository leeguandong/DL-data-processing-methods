# 调色板
# color_palette()能传入任何matplotlib所支持的颜色
# color_palette()不写参数则默认颜色
# set_palette()设置所有图的颜色

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

##################
# 连续性画板
current_palette = sns.color_palette()
sns.palplot(current_palette)
# 6个默认的颜色循环主题：deep muted pastel bright dark colorblind

# 圆形画板
# 当你有6个以上的分类要区分的时候，最简单的方法就是在一个圆形的颜色空间中画出均匀间隔的颜色
# 最常见的是使用hls的颜色空间，这是RGB值的一个简单转换
# 在画板中找出12个，一个圆盘的均匀12个
sns.palplot(sns.color_palette('hls', 12))

data = np.random.normal(size=(20, 8)) + np.arange(8) / 2
sns.boxplot(data=data, palette=sns.color_palette('hls', 8))
# l=亮度，s=饱和度

# 一对颜色，深浅有别
sns.palplot(sns.color_palette('Paired', 8))

# 使用xkcd颜色来命名颜色，xkcd字典存储了众多颜色及其命名
plt.plot([0, 1], [0, 1], sns.xkcd_rgb['pale red'], lw=3)
plt.plot([0, 1], [0, 2], sns.xkcd_rgb['medium green'], lw=3)

###################
# 连续性画板
# 色彩随数据变换，比如数据越来越重要则颜色越来越深
sns.palplot(sns.color_palette('Blues'))
# 想要翻转渐变，可以在面板名称中添加一个_r后缀
sns.palplot(sns.color_palette('BuGn_r'))

# cubehelix_palette()调色板
# 色调线性变换
sns.palplot(sns.color_palette('cubehelix', 8))
sns.palplot(sns.cubehelix_palette(8, start=.5, rot=-.75))

# light_palette()和dark_palette()调用定制连续调色板
sns.palplot(sns.light_palette('green'))
sns.palplot(sns.dark_palette('purple'))
sns.palplot(sns.light_palette('navy', reverse=True))

x, y = np.random.multivariate_normal([0, 0], [[1, -.5], [-.5, 1]], size=300).T
pal = sns.dark_palette('green', as_cmap=True)
sns.kdeplot(x, y, cmap=pal)

