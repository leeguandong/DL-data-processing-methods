# coding=utf-8
import numpy as np


# VOC-style mAP，分为两个计算方式，之所有两个计算方式，是因为2010年后VOC更新了评估方法，因此就有了07-metric和else...
def voc_ap(rec, prec, use_07_metric=False):
    """
    average precision calculations
    [precision integrated to recall]
    :param rec: recall list
    :param prec: precision list
    :param use_07_metric: 2007 metric is 11-recall-point based AP
    :return: average precision
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        # VOC07是11点插值的AP方式，等于是卡了11个离散的点，划分10个区间来计算AP
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0  # recall卡的阈值到顶了，1.1
            else:
                p = np.max(prec[rec >= t])  # VOC07：选择每个recall区间内对应的最高precision的计算方案
            ap = ap + p / 11.  # 11-recall-point based AP
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])  # 这个是不是动态规划？从后往前找之前区间内的top-precision，多么优雅的代码呀~~~

        # to calculate area under PR curve, look for points where X axis (recall) changes value
        # 上面的英文，可以结合着fig 2的绿框理解，一目了然
        # VOC10是是根据recall值变化的区间来计算的，如果recall变化很多次，就可以认为是一种 “伪” 连续的方式计算了，以下求的是recall的变化
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # 计算AP，这个计算方式有点玄乎，是一个积分公式的简化，应该是对应的fig 2中红色曲线以下的面积，之前公式的推导我有看过，现在有点忘了，麻烦各位同学补充一下
        # 现在理解了，不难，公式：sum (\Delta recall) * prec，其实结合fig2和下面的图，不就是算的积分么？如果recall划分得足够细，就可以当做连续数据，然后以下公式就是积分公式，算的precision、recall下面的面积了
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap
