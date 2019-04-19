# coding=utf-8
import numpy as np
import random
import cv2


def non_max_suppress(predicts_dict, threshold):
    # 对每一个类别分别进行NMS：一次读取一对键值(及某个类别的所有框)
    for object_name, bbox in predicts_dict.items():
        # 同一个键，对应多个不同的值，所以直接转成np.array处理了
        bbox_array = np.array(bbox, dtype=np.float)
        # 获取框左上角坐标(x1,y1),右下角坐标(x2,y2)以及此框的置信度
        x1 = bbox_array[:, 0]
        y1 = bbox_array[:, 1]
        x2 = bbox_array[:, 2]
        y2 = bbox_array[:, 3]
        scores = bbox_array[:, 4]
        # 同一个键对应的多组值下的score，按从大到小返回一个索引值
        order = scores.argsort()[::-1]
        # 当前类所有框的面积，即两矩阵对应元素相乘)；x1=3,x2=5,习惯上计算x方向长度就是x=3、4、5这三个像素，即5-3+1=3，而不是5-3=2，所以需要加1
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        keep = []

        #
        while order.size > 0:
            i = order[0]
            # 保留了当前最大confidence对应的bbox索引
            keep.append(i)
            # 获取所有与当前bbox的交集对应的左上角和右下角坐标，并计算IoU(这里是同时计算一个bbox与其他所有bbox的Iou)

            # 最大置信度的左上角坐标分别与剩余所有框的左上角坐标进行比较，分别保存较大值，因此这里的xx1的维数应该是当前类的框的个数减1
            # 所有的值和第一个值去比，以cls score为依据去选iou满足阈值的框
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0.0, xx2 - xx1 + 1) * np.maximum(0.0, yy2 - yy1 + 1)
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            # 保留iou小于等于阈值的索引值
            inds = np.where(iou <= threshold)
            # 将order中的第inds+1处的值重新赋值给order，即更新保留下来的索引，加1是因为没有计算与自身的IOU，所以索引相差1
            order = order[inds + 1]
        bbox = bbox_array[keep]
        predicts_dict[object_name] = bbox.tolist()
    return predicts_dict


# 下面在一张全黑图片上测试非极大抑制的效果
img = np.zeros((600, 600), np.uint8)
# 在全黑的图像上画出设定的几个框
predicts_dict = {'black1': [[83, 54, 165, 163, 0.8], [67, 48, 118, 132, 0.5], [91, 38, 192, 171, 0.6]],
                 'black2': [[59, 120, 137, 368, 0.12]]}
for object_name, bbox in predicts_dict.items():
    for box in bbox:
        x1, y1, x2, y2, score = box[0], box[1], box[2], box[3], box[-1]
        # uniform()方法将随机生成下一个实数，他在[x,y)范围内
        y_text = int(random.uniform(y1, y2))
        #
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2)
        #
        cv2.putText(img, str(score), (x2 - 30, y_text), 2, 1, (255, 255, 0))
    cv2.namedWindow('black_roi')
    cv2.imshow('black_roi', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 在全黑图片上画出经过极大值抑制后的框
img_cp = np.zeros((600, 600), np.uint8)
predicts_dict_nms = non_max_suppress(predicts_dict, 0.1)
for object_name, bbox in predicts_dict_nms.items():
    for box in bbox:
        x1, y1, x2, y2, score = int(box[0], box[1], box[2], box[3], box[-1])
        y_text = int(random.uniform(y1, y2))
        cv2.rectangle(img_cp, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.putText(img_cp, str(score), (x2 - 30, y_text), 2, 1, (255, 255, 0))
    cv2.namedWindow('black_nms')
    cv2.imshow('black_nms', img_cp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
