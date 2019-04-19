# coding=utf-8
import os
import random
import datetime
import re
import math
import logging
from collections import OrderedDict
import multiprocessing
import numpy as np
import tensorflow as tf
import keras
import keras.backend as K
import keras.layers as KL
import keras.engine as KE
import keras.models as KM
from MaskRCNN import utils

# Requires tensorflow.version import LooseVersion
from distutils.version import LooseVersion

assert LooseVersion(tf.__version__) >= LooseVersion('1.3')
assert LooseVersion(keras.__version__) >= LooseVersion('2.0.8')


# Utility functions
def log(text, array=None):
    pass


class BatchNorm(KL.BatchNormalization):
    pass


def compute_backbone_shapes(config, image_shape):
    pass


# Resnet Graph
def identity_block(input_tensor, kernel_size, filters, stage, block, use_bias=True, train_bn=True):
    pass


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2), use_bias=True, train_bn=True):
    pass


def resnet_graph(input_image, architecture, stage5=False, train_bn=True):
    pass


# Proposal layer
def apply_box_deltas_graph(boxes, deltas):
    '''
    Applies the given delta to the given boxes
    :param boxes: [N,(y1,x1,y2,x2)] boxes to update
    :param deltas:  [N,(dy,dx,log(dh),log(dw))] refinements to apply
    :return:
    '''
    # Convert to y,x,h,w
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    center_y = boxes[:, 0] + 0.5 * height
    center_x = boxes[:, 1] + 0.5 * width
    # Apply deltas
    center_y += deltas[:, 0] * height
    center_x += deltas[:, 1] * width
    height *= tf.exp(deltas[:, 2])
    width *= tf.exp(deltas[:, 3])
    # Convert back to y1,x1,y2,x2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    result = tf.stack([y1, x1, y2, x2], axis=1, name='apply_box_deltas_out')
    return result


def clip_boxes_graph(boxes, window):
    '''
    用于将超出图片范围的anchors进行剔除，这里由于回归框是归一化在[0,1]区间内，所以通过clip进行限定。
    :param boxes:[N,(y1,x1,y2,x2)]
    :param window: [4] in the form y1,x1,y2,x2   0,0,1,1
    :return:
    '''
    # Split
    wy1, wx1, wy2, wx2 = tf.split(window, 4)
    y1, x1, y2, x2 = tf.split(boxes, 4, axis=1)
    # TODO: print shape

    # Clip
    y1 = tf.maximum(tf.minimum(y1, wy2), wy1)
    x1 = tf.maximum(tf.minimum(x1, wx2), wx1)
    y2 = tf.maximum(tf.minimum(y2, wy2), wy1)
    x2 = tf.maximum(tf.minimum(x2, wx2), wx1)
    clipped = tf.concat([y1, x1, y2, x2], axis=1, name='clipped_boxes')
    clipped.set_shape((clipped.shape[0], 4))
    return clipped


class ProposalLayer(KE.Layer):
    '''
    Receives anchor scores and selects a subset to pass as proposals to the second stage.Filtering is done
    based on anchor scores and non-max suppression to remove overlaps.It also applies bounding box refinement
    deltas to anchors.

    Inputs:
      rpn_probs: [batch,num_anchors,(bg pro,fg prob)]
      rpn_bbox: [batch,num_anchors,(dy,dx,log(dh),log(dw))]
      anchors: [batch,num_anchors,(y1,x1,y2,x2)] anchors in normalized coordinates

    Returns:
         Proposals in normalized coordinates [batch,rois,(y1,x1,y2,x2)]
    '''

    def __init__(self, proposal_count, nms_threshold, config=None, **kwargs):
        super(ProposalLayer, self).__init__(**kwargs)
        self.config = config
        self.proposal_count = proposal_count
        self.nms_threshold = nms_threshold

    def call(self, inputs):
        # Box Scores,Use the foreground class confidence. [Batch,num_rois,1]
        scores = inputs[0][:, :, 1]
        # Box deltas [batch,num_rois,4]
        deltas = inputs[1]
        deltas = deltas * np.reshape(self.config.RPN_BBOX_STD_DEV, [1, 1, 4])
        # Anchors
        anchors = inputs[2]

        # Improve performance by trimming to top anchors by score and doing the rest on the smaller subset
        pre_nms_limit = tf.minimum(self.config.PRE_NMS_LIMIT, tf.shape(anchors)[1])
        ix = tf.nn.top_k(scores, pre_nms_limit, sorted=True, name='top_anchors').indices
        # 把前面选的从scores,deltas中切出来
        scores = utils.batch_slice([scores, ix], lambda x, y: tf.gather(x, y), self.config.IMAGES_PER_GPU)
        deltas = utils.batch_slice([deltas, ix], lambda x, y: tf.gather(x, y), self.config.IMAGES_PER_GPU)
        pre_nms_anchors = utils.batch_slice([anchors, ix], lambda a, x: tf.gather(a, x), self.config.IMAGES_PER_GPU,
                                            names=['pre_nms_anchors'])

        # Apply deltas to anchors to get refined anchors [batch,N,(y1,x1,y2,x2)]
        # 用 apply_box_deltas_out 函数使用deltas对anchor进行调整
        boxes = utils.batch_slice([pre_nms_anchors, deltas], lambda x, y: apply_box_deltas_graph(x, y),
                                  self.config.IMAGES_PER_GPU, names='refined_anchors')

        # clip to image boundaries. Since ew're in normalized coordinates,clip to 0..1 range,[batch,N,(y1,x1,y2,x2)]
        # window 是 0,0 到 1,1，超了就切掉了
        window = np.array([0, 0, 1, 1], dtype=np.float32)
        boxes = utils.batch_slice(boxes, lambda x: clip_boxes_graph(x, window), self.config.IMAGES_PER_GPU,
                                  names=['refined_anchors_clipped'])

        # Filter out small boxes

        # Non-max suppression
        def nms(boxes, score):
            indices = tf.image.non_max_suppression(boxes, scores, self.proposal_count, self.nms_threshold,
                                                   name='rpn_non_max_suppression')
            proposals = tf.gather(boxes, indices)
            # Pad if needed
            padding = tf.maximum(self.proposal_count - tf.shape(proposals)[0], 0)
            proposals = tf.pad(proposals, [(0, padding), (0, 0)])
            return proposals

        # TODO: proposals shape
        proposals = utils.batch_slice([boxes, scores], nms, self.config.IMAGES_PER_GPU)
        return proposals

    def compute_output_shape(self, input_shape):
        return (None, self.proposal_count, 4)


# ROIAlign Layer
def log2_graph(x):
    '''
    Implementation of log2
    :param x:
    :return:
    '''
    return tf.log(x) / tf.log(2.0)


class PyramidROIAlign(KE.Layer):
    '''
    Implements ROI pooling on multiple levels of the feature pyramid

    Params:
       pool_shape: [pool_height,pool_width] of the output pooled regions. Usually [7,7]

    Inputs:
       boxes: [batch,num_boxes,(y1,x1,y2,x2)] in normalized coordinates. Possibly padded with zeros if not enough boxes to fill the array.
       image_meta: [batch,(meta data)] Image details.See compose_image_meta()
       feature_maps: list of feature maps from different levels of the pyramid. Each is [batch,height,width,channel]

    output:
       Pooled regions in the shape: [batch,num_boxes,pool_height,pool_width,channel].The width and height are those
       specific in the pool_shape in the layer constructor.
    '''

    def __init__(self, pool_shape, **kwargs):
        super(PyramidROIAlign, self).__init__(**kwargs)
        self.pool_shape = tuple(pool_shape)

    def call(self, inputs):
        # crop boxes [batch,num_boxes,(y1,x1,y2,x2)] in normalized coords
        boxes = inputs[0]

        # Image meta
        # Hold details about the image. See compose_image_meta()
        image_meta = inputs[1]

        # Feature Maps. List of feature maps from different level of the feature pyramid.Each is [batch,height,width,channels]
        feature_maps = inputs[2:]

        # Assign each ROI to a level in the pyramid based on the ROI area
        y1, x1, y2, x2 = tf.split(boxes, 4, axis=2)
        h = y2 - y1
        w = x2 - x1
        # Use shape of first image.Image in a batch must have the same size
        image_shape = parse_image_meta_graph(image_meta)['image_shape'][0]
        # Equation 1 in the feature pyramid networks paper. Account for the fact that our coordinates are normzlized here
        # for example,a 224*224 ROI(in pixels) maps to P4
        image_area = tf.cast(image_shape[0] * image_shape[1], tf.float32)
        roi_level = log2_graph(tf.sqrt(h * w) / (224.0 / tf.sqrt(image_area)))
        roi_level = tf.minimum(5, tf.maximum(2, 4 + tf.cast(tf.round(roi_level), tf.int32)))
        roi_level = tf.squeeze(roi_level, 2)

        # Loop through levels and apply ROI pooling to each. P2 tO P5
        pooled = []
        box_to_level = []
        for i, level in enumerate(range(2, 6)):
            ix = tf.where(tf.equal(roi_level, level))
            level_boxes = tf.gather_nd(boxes, ix)

            # Box indices for crop_and_resize
            box_indices = tf.cast(ix[:, 0], tf.int32)

            # Keep track of which box is mapped to which level
            box_to_level.append(ix)

            # Stop gradient propogation to ROI proposals
            level_boxes = tf.stop_gradient(level_boxes)
            box_indices = tf.stop_gradient(box_indices)

            # Crop and resize
            # From mask rcnn paper:'We sample four regular locations,so that we can evaluate either max or average pooling
            # In fact, interpolating only a single value at each bin center(without pooling) is nearly effective.
            # Here we use the simplified approach of a single value per bin,which is how it's done in tf.crop_and_resize()
            # Result:[batch*num_boxes,pool_height,pool_width,channels]
            pooled.append(
                tf.image.crop_and_resize(feature_maps[i], level_boxes, box_indices, self.pool_shape, method='bilinear'))

        # Pack pooled features into one tensor
        pooled = tf.concat(pooled, axis=0)

        # Pack box_to_level mapping into one array and add another
        # column representing the order of pooled boxes
        box_to_level = tf.concat(box_to_level, axis=0)
        box_range = tf.expand_dims(tf.range(tf.shape(box_to_level)[0]), 1)
        box_to_level = tf.concat([tf.cast(box_to_level, tf.int32), box_range], axis=1)

        # Rearrange pooled features to match the order of the original boxes
        # Sort box_to_level by batch then box index
        # TF doesn't have a way to sort by columns,so merge them and sort
        sorting_tensor = box_to_level[:, 0] * 100000 + box_to_level[:, 1]
        ix = tf.nn.top_k(sorting_tensor, k=tf.shape(box_to_level)[0].indices[::-1])
        ix = tf.gather(box_to_level[:, 2], ix)
        pooled = tf.gather(pooled, ix)

        # Re-add the batch dimension
        shape = tf.concat([tf.shape(boxes)[:2], tf.shape(pooled)[1:]], axis=0)
        pooled = tf.reshape(pooled, shape)
        return pooled

    def compute_output_shape(self, input_shape):
        return input_shape[0][:2] + self.pool_shape + (input_shape[2][-1],)


# Detection Target Layer
def overlaps_graph(boxes1, boxes2):
    '''
     Computes IoU overlaps between two sets of boxes
    :param boxes1:  [N,(y1,x1,y2,x2)]
    :param boxes2:
    :return:
    '''
    # 1. Tile boxes and repeat boxes1. This allows us to compare every boxes1 against every boxes2 without loops
    # TF doesn't have an equivalent to np.repeat() so simulate it using tf.tile() and tf.reshape
    # TODO: b1,b2.shape
    b1 = tf.reshape(tf.tile(tf.expand_dims(boxes1, 1), [1, 1, tf.reshape(boxes2)[0]]), [-1, 4])
    b2 = tf.tile(boxes2, [tf.reshape(boxes1)[0], 1])
    # 2. Compute intersections
    b1_y1, b1_x1, b1_y2, b1_x2 = tf.split(b1, 4, axis=1)
    b2_y1, b2_x1, b2_y2, b2_x2 = tf.split(b2, 4, axis=1)
    y1 = tf.maximum(b1_y1, b2_y1)
    x1 = tf.maximum(b1_x1, b2_x1)
    y2 = tf.maximum(b1_y2, b2_y2)
    x2 = tf.maximum(b1_x2, b2_x2)
    intersection = tf.maximum(x2 - x1, 0) * tf.maximum(y2 - y1, 0)
    # 3. Compute unions
    b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
    b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
    union = b1_area + b2_area - intersection
    # 4. Compute IoU and reshape to [boxes1,boxes2]
    iou = intersection / union
    overlaps = tf.reshape(iou, [tf.shape(boxes1)[0], tf.shape(boxes2)[0]])
    return overlaps


def detection_targets_graph(proposals, gt_class_ids, gt_boxes, gt_masks, config):
    '''
    对一张图片生成检测目标。 Subsamples proposals and generates target class IDs,bounding box deltas and masks for each
    Inputs:
    :param proposals: [POST_NMS_ROIS_TRAINING,(y1,x1,y2,x2)] in normalized coordinates. proposals不充足就用0填充
    :param gt_class_ids: [MAX_GT_INSTANCES] int class IDs
    :param gt_boxes:  [MAX_GT_INSTANCES,(y1,x1,y2,x2)] in normalized coordinates
    :param gt_masks:  [height,width,MAX_GT_INSTANCES] of boolean type

    :return: Target ROIs and corresponding class IDs,bounding box shifts,and masks.
    rois:[TRAIN_ROIS_PER_IMAGE,(y1,x1,y2,x2)] in normalized coordinates
    class_ids:[TRAIN_ROIS_PER_IMAGE]  Integer class IDs. Zero padded
    deltas: [TRAIN_ROIS_PER_IMAGE,(dy,dw,log(dh),log(dw))]
    masks: [TRAIN_ROIS_PER_IMAGE,heights,width].Masks cropped to bbox boundaries and resized to neural network output size.

    Note:Returned arrays might be zero padded if not enough target ROIs
    '''
    # Assertions
    asserts = [tf.Assert(tf.greater(tf.reshape(proposals)[0], 0), [proposals], name='roi_assertion')]
    with tf.control_dependencies(asserts):
        proposals = tf.identity(proposals)

    # Remove zero padding 把0在数据里去掉
    proposals, _ = trim_zeros_graph(proposals, name='trim_proposals')
    gt_boxes, non_zeros = trim_zeros_graph(gt_boxes, name='trim_gt_boxes')
    # 在gt_class_ids里面选非边框为0的类
    gt_class_ids = tf.boolean_mask(gt_class_ids, non_zeros, name='trim_gt_class_ids')
    gt_masks = tf.gather(gt_masks, tf.where(non_zeros)[:, 0], axis=2, name='trim_gt_masks')

    # Handle COCO crowds
    # A crowd box in COCO is a bounding box around several instances.Exclude them from training. A crowd box is given a negative class ID
    crowd_ix = tf.where(gt_class_ids < 0)[:, 0]
    non_crowd_ix = tf.where(gt_class_ids > 0)[:, 0]
    crowd_boxes = tf.gather(gt_boxes, crowd_ix)
    crowd_masks = tf.gather(gt_masks, crowd_ix, axis=2)
    gt_class_ids = tf.gather(gt_class_ids, non_crowd_ix)
    gt_boxes = tf.gather(gt_boxes, non_crowd_ix)
    gt_masks = tf.gather(gt_masks, non_crowd_ix, axis=2)

    # Compute overlaps matrix [proposals,gt_boxes]
    overlaps = overlaps_graph(proposals, gt_boxes)

    # Compute overlaps with crowd boxes [proposals,crowd_boxes]
    crowd_overlaps = overlaps_graph(proposals, crowd_boxes)
    crowd_iou_max = tf.reduce_max(crowd_overlaps, axis=1)
    no_crowd_bool = (crowd_iou_max < 0.001)

    # Determine positive and negative ROIs
    roi_iou_max = tf.reduce_max(overlaps, axis=1)
    # 1. Positive ROIs are those with >= 0.5 IoU with a GT box
    positive_roi_bool = (roi_iou_max >= 0.5)
    positive_indices = tf.where(positive_roi_bool)[:, 0]
    # 2. Negative ROIs are those with < 0.5 with every GT box.skip crowd
    negative_indices = tf.where(tf.logical_and(roi_iou_max < 0.5, no_crowd_bool))[:, 0]

    # Subsample ROIs. Aim for 33% positive
    # Positive ROIs
    positive_count = int(config.TRAIN_ROIS_PER_IMAGE * config.ROI_POSITIVE_RATIO)
    positive_indices = tf.random_shuffle(positive_indices)[:positive_count]
    positive_count = tf.shape(positive_indices)[0]
    # Negative ROIs.Add enough to maintain positive:negative ratio
    r = 1.0 / config.ROI_POSITIVE_RATIO
    negative_count = tf.cast(r * tf.cast(positive_count, tf.float32), tf.int32) - positive_count
    negative_indices = tf.random_shuffle(negative_indices)[:negative_count]
    # Gather selected ROIs
    positive_rois = tf.gather(proposals, positive_indices)
    negative_rois = tf.gather(proposals, negative_indices)

    # Assign positive ROIs to GT boxes
    positive_overlaps = tf.gather(overlaps, positive_indices)
    roi_gt_box_assignment = tf.cond(tf.gather(tf.shape(positive_overlaps)[1], 0),
                                    true_fn=lambda: tf.argmax(positive_overlaps, axis=1),
                                    false_fn=lambda: tf.cast(tf.constant([]), tf.int64))
    roi_gt_boxes = tf.gather(gt_boxes, roi_gt_box_assignment)
    roi_gt_class_ids = tf.gather(gt_class_ids, roi_gt_box_assignment)

    # Compute bbox refinement for positive ROIs
    deltas = utils.box_refinement_graph(positive_rois, roi_gt_boxes)
    deltas /= config.BBOX_STD_DEV

    # Assign positive ROIs to GT masks
    # Permute masks to [N,height,width,1]
    transposed_masks = tf.expand_dims(tf.transpose(gt_masks, [2, 0, 1]), -1)
    # Pick the right mask for each ROI
    roi_masks = tf.gather(transposed_masks, roi_gt_box_assignment)

    # Compute mask targets 归一化操作
    boxes = positive_rois
    if config.USE_MINI_MASK:
        # Transform ROI coordinates from normalized image space to normalized mini-mask space
        y1, x1, y2, x2 = tf.split(positive_rois, 4, axis=1)
        gt_y1, gt_x1, gt_y2, gt_x2 = tf.split(roi_gt_boxes, 4, axis=1)
        gt_h = gt_y2 - gt_y1
        gt_w = gt_x2 - gt_x1
        y1 = (y1 - gt_y1) / gt_h
        y2 = (x1 - gt_x1) / gt_w
        y2 = (y2 - gt_y1) / gt_h
        x2 = (x2 - gt_x1) / gt_w
        boxes = tf.concat([y1, x1, y2, x2], 1)
    box_ids = tf.range(0, tf.reshape(roi_masks)[0])
    masks = tf.image.crop_and_resize(tf.cast(roi_masks, tf.float32), boxes, box_ids, config.MASK_SHAPE)

    # Remove the extra dimension from masks
    masks = tf.squeeze(masks, axis=3)

    # Threshold mask pixels at 0.5 to have GT masks be 0 or 1 to use with binary cross entropy loss
    masks = tf.round(masks)

    # Append negative ROIs and pad bbox deltas and masks that are not used for negative ROIs with zeros
    rois = tf.concat([proposals, negative_rois], axis=0)
    N = tf.reshape(negative_rois)[0]
    P = tf.maximum(config.TRAIN_ROIS_PER_IMAGE - tf.shape(rois)[0], 0)
    rois = tf.pad(rois, [(0, P), (0, 0)])
    roi_gt_boxes = tf.pad(roi_gt_boxes, [(0, N + P), (0, 0)])
    roi_gt_class_ids = tf.pad(roi_gt_class_ids, [(0, N + P)])
    deltas = tf.pad(deltas, [(0, N + P), (0, 0)])
    masks = tf.pad(masks, [[0, N + P], (0, 0), (0, 0)])
    return rois, roi_gt_class_ids, deltas, masks


class DetectionTargetLayer(KE.Layer):
    '''
    Subsamples proposals and generates target box refinement,class_ids,and masks for each

    Inputs:
    proposals: [batch,N,(y1,x1,y2,x2)] in normalized coordinates.Might be zero padded if there are not enough proposals
    gt_class_ids: [batch,MAX_GT_INSTANCES] Interger class IDs
    gt_boxes: [batch,MAX_GT_INSTANCES,(y1,x1,y2,x2)] in normalized coordinates
    gt_masks:[batch,height,width,MAX_GT_INSTANCES] of boolean type

    Returns: Target ROIs and corresponding class IDs,bounding box shifts,and masks.
    rois:[batch,TRAIN_ROIS_PER_IMAGE,(y1,x1,y2,x2)] in normalized coordinates
    target_class_ids: [batch,TRAIN_ROIS_PER_IMAGE] Integer class IDs.
    target_deltas: [batch,TRAIN_ROIS_PER_IMAGE,(dy,dx,log(dh),log(dw))]
    target_mask:[batch,TRAIN_ROIS_PER_IMAGE,height,width] MASK cropped to bbox boundaries and resized to neural network output size
    '''

    def __init__(self, config, **kwargs):
        super(DetectionTargetLayer, self).__init__(**kwargs)
        self.config = config

    def call(self, inputs):
        # rois,是个list，后面全是标签
        proposals = inputs[0]
        gt_class_ids = inputs[1]
        gt_boxes = inputs[2]
        gt_masks = inputs[3]

        # Slice the batch and run a graph for each slice  gt是标签，target是检测结果
        # TODO: Rename target_bbox to target_deltas for clarity
        names = ['rois', 'target_class_ids', 'target_bbox', 'target_mask']
        outputs = utils.batch_slice([proposals, gt_class_ids, gt_boxes, gt_masks],
                                    lambda w, x, y, z: detection_targets_graph(w, x, y, z, self.config),
                                    self.config.IMAGES_PER_GPU, names=names)
        return outputs

    def compute_output_shape(self, input_shape):
        return [(None, self.config.TRAIN_ROIS_PER_IMAGE, 4),  # rois
                (None, self.config.TRAIN_ROIS_PER_IMAGE),  # class_ids
                (None, self.config.TRAIN_ROIS_PER_IMAGE, 4),  # deltas
                (None, self.config.TRAIN_ROIS_PER_IMAGE, self.config.MASK_SHAPE[0])
                ]

    def compute_mask(self, inputs, mask=None):
        return [None, None, None, None]


# Detection Layer
def refine_detections_graph(rois, probs, deltas, window, config):
    '''
    Refine classified proposals and filter overlaps and return final detections
    a. 获取每个推荐区域得分最高的class的得分
    b. 获取每个推荐区域经过粗修后的坐标和window交集的坐标
    c. 剔除掉最高分为背景的推荐区域
    d. 剔除掉最高得分达不到阈值的推荐区域
    e. 对属于同一类别的候选框进行非极大值抑制
    f. 对非极大值抑制后的框索引：剔除-1占位符，获取top k
    Inputs:
    :param rois: [N,(y1,x1,y2,x2)] in normalized coordinates
    :param probs: [N,num_classes]. class probabilities
    :param deltas: [N,num_classes,(dy,dx,log(dh),log(dw))].class-specific bounding box deltas
    :param window: (y1,x1,y2,x2) in normalized coordinates. The part of the image that contains the image excluding the padding
    :param config:
    :return:
       detections shaped: [num_detections,(y1,x1,y2,x2,class_id,score)] where coordinated are normalized
    '''
    # Class IDs per ROI
    class_ids = tf.argmax(probs, axis=1, output_type=tf.int32)
    # Class probability of the top class of each ROI
    indices = tf.stack([tf.range(probs.shape[0]), class_ids], axis=1)
    class_scores = tf.gather_nd(probs, indices)
    # class-specific bounding box deltas
    deltas_specific = tf.gather_nd(deltas, indices)
    # Apply bounding box deltas
    # shape:[boxes,(y1,x1,y2,x2)] in normalized coordinates
    refined_rois = apply_box_deltas_graph(rois, deltas_specific * config.BBOX_STD_DEV)
    # Clip boxes to image window
    refined_rois = clip_boxes_graph(refined_rois, window)

    # TODO:Filter out boxes with zero area

    # Filter out background boxes
    keep = tf.where(class_ids > 0)[:, 0]
    # Filter out low confidence boxes.
    if config.DETECTION_MIN_CONFIDENCE:
        conf_keep = tf.where(class_scores >= config.DETECTION_MIN_CONFIDENCE)[:, 0]
        keep = tf.sets.set_intersection(tf.expand_dims(keep, 0), tf.expand_dims(conf_keep, 0))
        keep = tf.sparse_tensor_to_dense(keep)[0]

    # Apply per-class NMS
    # 1.Prepare variables
    pre_nms_class_ids = tf.gather(class_ids, keep)
    pre_nms_score = tf.gather(class_scores, keep)
    pre_nms_rois = tf.gather(refined_rois, keep)
    unique_pre_nms_class_ids = tf.unique(pre_nms_class_ids)[0]

    def nms_keep_map(class_id):
        '''
        Apply non-maximum suppression on rois of the given class
        :param class_id:
        :return:
        '''
        # Indices of ROIs of the given class
        ixs = tf.where(tf.equal(pre_nms_class_ids, class_id))[:, 0]
        # Apply NMS
        class_keep = tf.image.non_max_suppression(tf.gather(pre_nms_rois, ixs), tf.gather(pre_nms_score, ixs),
                                                  max_output_size=config.DETECTION_MAX_INSTANCES,
                                                  iou_threshold=config.DETECTION_NMS_THRESHOLD)
        # Map indices
        class_keep = tf.gather(keep, tf.gather(ixs, class_keep))
        # Pad with -1 so returned tensors have the same shape
        gap = config.DETECTION_MAX_INSTANCES - tf.shape(class_keep)[0]
        class_keep = tf.pad(class_keep, [(0, gap)], mode='CONSTANT', constant_values=-1)
        # Set shape so map_fn() can infer result shape
        class_keep.set_shape([config.DETECTION_MAX_INSTANCES])
        return class_keep

    # 2. Map over class IDs
    nms_keep = tf.map_fn(nms_keep_map, unique_pre_nms_class_ids, dtype=tf.int32)
    # 3. Merge results into one list,and remove -1 padding
    nms_keep = tf.reshape(nms_keep, [-1])
    nms_keep = tf.gather(nms_keep, tf.where(nms_keep > -1)[:, 0])
    # 4. Compute intersection between keep and nms_keep
    keep = tf.sets.set_intersection(tf.expand_dims(keep, 0), tf.expand_dims(nms_keep, 0))
    keep = tf.sparse_tensor_to_dense(keep)[0]
    # Keep top detection
    roi_count = config.DETECTION_MAX_INSTANCES
    class_scores_keep = tf.gather(class_scores, keep)
    num_keep = tf.minimum(tf.shape(class_scores_keep)[0], roi_count)
    top_ids = tf.nn.top_k(class_scores_keep, k=num_keep, sorted=True)[1]
    keep = tf.gather(keep, top_ids)

    # Arrange output as [N,(y1,x1,y2,x2,class_id,score)]
    # Coordinates are normalized
    detections = tf.concat([tf.gather(refined_rois, keep), tf.to_float(tf.gather(class_ids, keep))[..., tf.newaxis],
                            tf.gather(class_scores, keep)[..., tf.newaxis]], axis=1)

    # Pad with zeros if detections < DETECTION_MAX_INSTANCES
    gap = config.DETECTION_MAX_INSTANCES - tf.shape(detections)[0]
    detections = tf.pad(detections, [(0, gap), (0, 0)], 'CONSTANT')
    return detections


class DetectionLayer(KE.Layer):
    '''
    Takes classified proposal boxes and their bounding box deltas and returns the final detection boxes

    '''

    def __init__(self, config=None, **kwargs):
        super(DetectionLayer, self).__init__(**kwargs)
        self.config = config

    def call(self, inputs):
        rois = inputs[0]
        mrcnn_class = inputs[1]
        mrcnn_bbox = inputs[2]
        image_meta = inputs[3]

        # 获取window参数即原始图片大小，然后获取其相对于输入图片的image_shape即[w,h,channels]的尺寸
        # Get windows of image in normalized coordinates.Windows are the area in the image that excludes the padding
        # Use the shape of the first image in the batch to normalize the window because we know that all images get resized to the same size
        # 用于解析并获取输入图片的shape和原始图片的shape(window)
        m = parse_image_meta_graph(image_meta)
        image_shape = m['image_shape'][0]
        window = norm_boxes_fraph(m['window'], image_shape[:2])

        # Run detection refinement graph on each item in the batch
        detections_batch = utils.batch_slice([rois, mrcnn_class, mrcnn_bbox, window],
                                             lambda x, y, w, z: refine_detections_graph(x, y, w, z, self.config),
                                             self.config.IMAGES_PER_GPU)

        # Reshape output
        # [batch,num_detections,(y1,x1,y2,x2,class_id,class_score)] in normalized coordinates
        return tf.reshape(detections_batch, [self.config.BATCH_SIZE, self.config.DETECTION_MAX_INSTANCES, 6])

    def compute_output_shape(self, input_shape):
        return (None, self.config.DETECTION_MAX_INSTANCES, 6)


# Region Proposal netwrok(RPN)
def rpn_graph(feature_map, anchor_per_location, anchor_stride):
    '''
    builds the computation graph of region proposal network

    :param feature_map: backbone features [batch,height,width,depth]
    :param anchor_per_location: number of anchors per pixel in the feature map,3
    :param anchor_stride: controls the density of anchors.Typically 1 or 2
    :return:
        rpn_class_logits: [batch,H*W*anchors_per_location,2] Anchor classifier logits (before softmax)
        rpn_probs: [batch,H*W*anchor_per_location,2] Anchor classifier probabilites
        rpn_bbox: [batch,H*W*anchors_per_location,(dy,dx,log(dh),log(dw))] Deltas to be applied to anchors.
    '''
    # TODO:check if stride of 2 causes alignment issues if the feature map is not even
    # shared convolutional base of the RPN
    shared = KL.Conv2D(512, (3, 3), padding='same', activation='relu', strides=anchor_stride, name='rpn_conv_shared')(
        feature_map)

    # Anchor score. [batch,height,width,anchors per location * 2]
    x = KL.Conv2D(2 * anchor_per_location, (1, 1), padding='valid', activation='linear', name='rpn_class_raw')(shared)

    # Reshape to [batch,anchors,2]
    rpn_class_logits = KL.Lambda(lambda t: tf.reshape(t, [tf.reshape(t)[0], -1, 2]))(x)

    # Softmax on least dimension of BG/FG
    rpn_probs = KL.Activation('softmax', name='rpn_class')(rpn_class_logits)

    # Bounding Box refinement. [batch,H,W,anchor per location * depth] 每个像素点三个anchor，每层上都会产生anchor
    # where depth is [x,y,log(w),log(h)]
    x = KL.Conv2D(anchor_per_location * 4, (1, 1), padding='valid', activation='linear', name='rpn_bbox_pred')(shared)

    # Reshape to [batch,anchors,4]
    rpn_bbox = KL.Lambda(lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 4]))(x)

    return [rpn_class_logits, rpn_probs, rpn_bbox]


def build_rpn_model(anchor_stride, anchors_per_location, depth):
    '''
    Builds a keras model of the Region proposal network
    It wraps the RPN graph so it can be used multiple times with shared weights

    :param anchor_stride: controls the density of anchors.Typically 1(anchors for every pixel in the feature map),or 2(every other pixel)
    :param anchors_per_location: number of anchors per pixel in the feature map
    :param depth: depth of the backbone feature map
    :return:
    rpn_class_logits: [batch,H*W*anchors_per_location,2] Anchor classifier logits (before softmax)
    rpn_probs: [batch,H*W*anchor_per_location,2] Anchor classifier probabilites
    rpn_bbox: [batch,H*W*anchors_per_location,(dy,dx,log(dh),log(dw))] Deltas to be applied to anchors.
    '''
    input_feature_map = KL.Input(shape=[None, None, depth], name='input_rpn_feature_map')
    outputs = rpn_graph(input_feature_map, anchors_per_location, anchor_stride)
    return KM.Model([input_feature_map], outputs, name='rpn_model')


# Feature Pyramid network heads
def fpn_classifier_graph(rois, feature_maps, image_meta, pool_size, num_classes, train_bn=True, fc_layers_size=1024):
    '''
    Builds the computation graph of the feature pyramid network classifier and regression heads
    :param rois: [batch,num_rois,(y1,x1,y2,x2)] Proposal boxes in normalized coordinates
    :param feature_maps:  List of feature maps from different layers of the pyramid,[P2,p3,p4,p5]. Each has a different resolution
    :param image_meta: [batch,(meta data)] Image details.See compose_image_meta()
    :param pool_size:  The width of the square feature map generated from ROI pooling
    :param num_classes: number of classes,which determines the depth of the results
    :param train_bn:  boolean.Train of freeze Batch Norm layer
    :param fc_layers_size:  size of the 2 fc layers
    :return:
    logits: [batch,num_rois,NUM_CLASSES] classifier logits  (before softmax)
    probs: [batch,num_rois,NUM_CLASSES] classifier probabilities
    bbox_delats:[batch,num_rois,NUM_CLASSES,(dy,dx,log(dh),log(dw)] Deltas to apply to proposal boxes
    '''
    # ROI pooling
    # Shape:[batch,num_rois,POOL_SIZE,POOL_SIZE,channels]
    x = PyramidROIAlign([pool_size, pool_size], name='roi_align_classifier')([rois, image_meta] + feature_maps)
    # Two 1024 FC layers(implemented with Conv2D for consistency)
    x = KL.TimeDistributed(KL.Conv2D(fc_layers_size, (pool_size, pool_size), padding='valid'),
                           name='mrcnn_class_conv1')(x)
    x = KL.TimeDistributed(BatchNorm(), name='mrcnn_class_bn1')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    x = KL.TimeDistributed(KL.Conv2D(fc_layers_size, (1, 1)), name='mrcnn_class_conv2')(x)
    x = KL.TimeDistributed(BatchNorm(), name='mrcnn_class_bn2')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    shared = KL.Lambda(lambda x: K.squeeze(K.squeeze(x, 3), 2), name='pool_squeeze')(x)

    # Classifier head
    mrcnn_class_logits = KL.TimeDistributed(KL.Dense(num_classes), name='mrcnn_class_logits')(shared)
    mrcnn_probs = KL.TimeDistributed(KL.Activation('softmax'), name='mrcnn_class')(mrcnn_class_logits)

    # BBox head
    # [batch,num_rois,NUM_CLASSES*(dy,dx,log(dh),log(dw))]
    x = KL.TimeDistributed(KL.Dense(num_classes * 4, activation='linear'), name='mrcnn_bbox_fc')(shared)
    # Reshape to [batch,num_rois,NUM_CLASSES,(dy,dx,log(dh),log(dw))]
    s = K.int_shape(x)
    mrcnn_bbox = KL.Reshape((s[1], num_classes, 4), name='mrrcnn_box')(x)
    return mrcnn_class_logits, mrcnn_probs, mrcnn_bbox


def build_fpn_mask_graph(rois, feature_maps, image_meta, pool_size, num_classes, train_bn=True):
    '''
    Builds the computation graph of the mask head of Feature pyramid network
    :param rois:  [batch,num_rois,(y1,x1,y2,x2)] proposal boxes in normalized coordinates
    :param feature_maps: List of feature maps from different layers of thepyramid,[P2,P3,P4,P5].Each has a different resolution
    :param image_meta: [batch,(meta data)] Image details.See compose_image_meta()
    :param pool_size: The width of the square feature map generated from ROI Pooling
    :param num_classes: number of classes,which determines the depth of the results
    :param train_bn: Boolean. Train or freeze Batch Norm layers
    :return:
    Masks [batch,num_rois,MASK_POOL_SIZE,MASK_POOL_SIZE,NUM_CLASSES]
    '''
    # ROI Pooling
    # Shape:[batch,num_rois,MASK_POOL_SIZE,MASK_POOL_SIZE,channels]
    x = PyramidROIAlign([pool_size, pool_size], name='roi_align_mask')([rois, image_meta] + feature_maps)

    # Conv layers
    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding='same'), name='mrcnn_mask_conv1')(x)
    x = KL.TimeDistributed(BatchNorm(), name='mrcnn_mask_bn1')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding='same'), name='mrcnn_mask_conv2')(x)
    x = KL.TimeDistributed(BatchNorm(), name='mrcnn_mask_bn2')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding='same'), name='mrcnn_mask_conv3')(x)
    x = KL.TimeDistributed(BatchNorm(), name='mrcnn_mask_bn3')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding='same'), name='mrcnn_mask_conv4')(x)
    x = KL.TimeDistributed(BatchNorm(), name='mrcnn_mask_bn4')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2DTranspose(256, (2, 2), strides=2, activation='relu'), name='mrcnn_mask_deconv')(x)
    x = KL.TimeDistributed(KL.Conv2D(num_classes, (1, 1), strides=1, activation='sigmoid'), name='mrcnn_mask')(x)
    return x


# Loss Functions
def smooth_l1_loss(y_true, y_pred):
    '''
    Implements smooth-l1 loss
    y_true and y_pred are typically: [N,4],but could be any shape
    :param x_true:
    :param y_pred:
    :return:
    '''
    diff = K.abs(y_true - y_pred)
    less_than_one = K.cast(K.less(diff, 1.0), 'float32')
    loss = (less_than_one * 0.5 * diff ** 2) + (1 - less_than_one) * (diff - 0.5)
    return loss


def rpn_class_loss_graph(rpn_match, rpn_class_logits):
    '''
    RPN anchor classifier loss
    :param rpn_match: [batch,anchors,1].Anchor match type,1=positive,-1=negative,0=neyral anchor
    :param rpn_class_logits: [batch,anchors,2]. RPN classifier logits for FG/BG
    :return:
    '''
    # Squeeze last dim to simplify
    rpn_match = tf.squeeze(rpn_match, -1)
    # Get anchor classes. Convert the -1/+1 match to 0/1 values
    anchor_class = K.cast(K.equal(rpn_match, 1), tf.int32)
    # Positive and Negative anchors contribute to the loss,but neutral anchors(match value = 0) don't
    indices = tf.where(K.not_equal(rpn_match, 0))
    # Pick rows that contribute to the loss and filter out the rest
    rpn_class_logits = tf.gather_nd(rpn_class_logits, indices)
    anchor_class = tf.gather_nd(anchor_class, indices)
    # cross entropy loss
    loss = K.sparse_categorical_crossentropy(target=anchor_class, output=rpn_class_logits, from_logits=True)
    loss = K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0))
    return loss


def rpn_bbox_loss_graph(config, target_bbox, rpn_match, rpn_bbox):
    '''
    Return the RPN bounding box loss graph
    :param config:  the model config object
    :param target_bbox: [batch,max positive anchors,(dy,dx,log(dh),log(dw))]. Use 0 padding to fill in unsed bbox deltas.
    :param rpn_match: [batch,anchors,1].Anchor match type. 1=positive,-1=negative,0=neutral anchor
    :param rpn_bbox: [batch,anchors,(dy,dx,log(dh),log(dw))]
    :return:
    '''
    # positive anchors contribute to the loss,but negative and neutral anchors (match value of 0 or -1) don't
    rpn_match = K.squeeze(rpn_match, -1)
    indices = tf.where(K.equal(rpn_match, 1))
    # pick bbox deltas that contribute to the loss
    rpn_bbox = tf.gather_nd(rpn_bbox, indices)

    # Trim target bounding box deltas to the same length as rpn_bbox
    batch_counts = K.sum(K.cast(K.equal(rpn_match, 1), tf.int32), axis=1)
    target_bbox = batch_pack_graph(target_bbox, batch_counts, config.IMAGE_PER_GPU)
    loss = smooth_l1_loss(target_bbox, rpn_bbox)
    loss = K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0))
    return loss


def mrcnn_class_loss_graph(target_class_ids, pred_class_logits, active_class_ids):
    '''
    Loss for the classifier head of Mask RCNN
    :param target_class_ids: [batch,num_rois].Integer class IDs.Uses zero padding to fill in the array
    :param pred_class_logits: [batch,num_rois,num_classes]
    :param active_class: [batch,num_classes].Has a value of 1 for classes that are in the dataset of the image,and 0 for classes that are not in the dataset
    :param ids:
    :return:
    '''
    # During model building,keras calls this function with target_class_ids of type float32. Unclear why,Cast it to int to get around it
    target_class_ids = tf.cast(target_class_ids, 'int64')

    # Find predictions of classes that are not in the dataset
    pred_class_ids = tf.argmax(pred_class_logits, axis=2)
    # TODO:
    pred_active = tf.gather(active_class_ids[0], pred_class_ids)

    # Loss
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_class_ids, logits=pred_class_logits)
    # Earse losses of predictions of classes that are not in the active classes of the image
    loss = loss * pred_active

    # Computer loss mean. Use only predictions that contribute to the loss to get a correct mean
    loss = tf.reduce_sum(loss) / tf.reduce_sum(pred_active)
    return loss


def mrcnn_bbox_loss_graph(target_bbox, target_class_ids, pred_bbox):
    '''
    Loss for the classifier head of Mask RCNN
    :param target_bbox: [batch,num_rois,(dy,dx,log(dh),log(dw))]
    :param target_class_ids: [batch,num_rois]. Integer class IDs
    :param pred_bbox: [batch,num_rois,num_classes,(dy,dx,log(dh),log(dw))]
    :return:
    '''
    # Reshape to merge batch and roi dimensions for simplicity
    target_class_ids = K.reshape(target_class_ids, (-1,))
    target_bbox = K.reshape(target_bbox, (-1, 4))
    pred_bbox = K.reshape(pred_bbox, (-1, K.int_shape(pred_bbox)[2], 4))

    # Only positive ROIs contribute to the loss.And only the right class_id of each ROI.Get their indices
    positive_roi_ix = tf.where(target_class_ids > 0)[:, 0]
    positive_roi_class_ids = tf.cast(tf.gather(target_class_ids, positive_roi_ix), tf.int64)
    indices = tf.stack([positive_roi_ix, positive_roi_class_ids], axis=1)

    # Gather the deltas (predicted and true) that contribute to loss
    target_bbox = tf.gather(target_bbox, positive_roi_ix)
    pred_bbox = tf.gather_nd(pred_bbox, indices)

    # Smooth-L1 loss
    loss = K.switch(tf.size(target_bbox) > 0, smooth_l1_loss(y_true=target_bbox, y_pred=pred_bbox), tf.constant(0.0))
    loss = K.mean(loss)
    return loss


def mrcnn_mask_loss_graph(target_masks, target_class_ids, pred_masks):
    '''
    Mask binary cross-entropy loss for the masks head
    :param target_masks:  [batch,num_rois,height,width]. A float32 tensor of values 0 or 1.Uses zero padding to fill array
    :param target_class_ids:  [batch,num_rois].Integer class IDs.zero padded.
    :param pred_masks:  [batch,proposals,height,width,num_classes] float32 tensor with values from 0 to 1
    :return:
    '''
    # Reshape for simplicity.Merge first two dimensions into one
    target_class_ids = K.reshape(target_class_ids, (-1,))
    mask_shape = tf.shape(target_masks)
    target_masks = K.reshape(target_masks, (-1, mask_shape[2], mask_shape[3]))
    pred_shape = tf.shape(pred_masks)
    pred_masks = K.reshape(pred_masks, (-1, pred_shape[2], pred_shape[3], pred_shape[4]))
    # Permute predicted masks to [N,num_classes,height,width]
    pred_masks = tf.transpose(pred_masks, [0, 3, 1, 2])

    # Only positive ROIs contribute to the loss.And only the class specific mask of each ROI
    positve_ix = tf.where(target_class_ids > 0)[:, 0]
    positve_class_ids = tf.cast(tf.gather(target_class_ids, positve_ix), tf.int64)
    indices = tf.stack([positve_ix, positve_class_ids], axis=1)

    # Gather the masks (predicted and true) that contribute to loss
    y_true = tf.gather(target_masks, positve_ix)
    y_pred = tf.gather(pred_masks, indices)

    # Compute binary cross entropy. If no positive ROIs,then return 0
    # shape:[batch,roi,num_classes]
    loss = K.switch(tf.size(y_true) > 0, K.binary_crossentropy(target=y_true, output=y_pred), tf.constant(0.0))
    loss = K.mean(loss)
    return loss


# Data generator
def load_image_gt(dataset, config, image_id, augment=False, augmentation=None, use_mini_mask=False):
    pass


def build_detection_targets(rpn_rois, gt_class_ids, gt_bbox, gt_masks, config):
    pass


def build_rpn_targets(image_shape, anchors, gt_class_ids, gt_boxes, config):
    pass


def generate_random_rois(image_shape, count, gt_class_ids, gt_boxes):
    pass


def data_generator(dataset, config, shuffle=True, augment=False, augmentation=None, random_rois=0, batch_size=1,
                   detection_target=False, no_augmentation_sources=None):
    pass


# MaskRCNN Class
class MaskRCNN():
    '''概述 MaskRcnn的函数'''

    def __init__(self, mode, config, model_dir):
        '''
        :param mode:  training and inference
        :param config:
        :param model_dir:  directory to save training logs and trained weights
        '''
        assert mode in ['training', 'inference']
        self.mode = mode
        self.config = config
        self.model_dir = model_dir
        self.set_log_dir()
        self.keras_model = self.build(mode=mode, config=config)

    def build(self, mode, config):
        '''
        Build maskrcnn architecture
        :param mode: training or inference.the inputs and outputs of the model differ accordingly
        :param config:
        :return:
        '''
        assert mode in ['training', 'inference']

        # Image size must be dividable by 2 multiple times
        # 图像大小需要可以被2整除多次，至少6次
        h, w = config.IMAGE_SHAPE[:2]
        if h / 2 ** 6 != int(h / 2 ** 6) or w / 2 ** 6 != int(w / 2 ** 6):
            raise Exception('Image size must be dividable by 2 at least 6 times'
                            'to avoid fractions when downscaling and upscaling.'
                            'for example,use 256,320,384,448,512...etc')

        # Inputs
        input_image = KL.Input(shape=[None, None, config.IMAGE_SHAPE[2]], name='input_image')
        input_image_meta = KL.Input(shape=[config.IMAGE_SHAPE_SIZE], name='input_image_meta')

        if mode == 'training':
            # 从特征提取出来有两路，一条是rpn，一条是feature map
            # RPN GT, match == score, bbox == 坐标
            input_rpn_match = KL.Input(shape=[None, 1], name='input_rpn_match', dtype=tf.int32)
            input_rpn_bbox = KL.Input(shape=[None, 4], name='input_rpn_bbox', dtype=tf.float32)

            # Detection GT (class IDs, bounding boxes, and masks)
            # 1.GT classes IDs(zero padded)
            input_gt_class_ids = KL.Input(shape=[None], name='input_gt_class_ids', dtype=tf.int32)
            # 2.GT Boxes in pixels(zero padded)
            # [batch, MAX_GT_INSTANCES, (y1,x1,y2,x2)] in image coordinates
            input_gt_boxes = KL.Input(shape=[None, 4], name='input_gt_boxes', dtype=tf.float32)
            # Normalize coordinates
            gt_boxes = KL.Lambda(lambda x: norm_boxes_fraph(x, K.shape(input_image)[1:3]))(input_gt_boxes)
            # 3.GT Masks(zero padded)
            # [batch, height, width, MAX_GT_INSTANCES]
            if config.USE_MINI_MASK:
                input_gt_masks = KL.Input(shape=[config.MINI_MASK_SHAPE[0], config.MINI_MASK_SHAPE[1], None],
                                          name='input_gt_masks', dtype=bool)
            else:
                input_gt_boxes = KL.Input(shape=[config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1], None],
                                          name='input_gt_masks', dtype=bool)
        elif mode == 'inference':
            # Anchors in normalized coordinates
            input_anchors = KL.Input(shape=[None, 4], name='input_anchors')

        # Build the shared convolutional layers
        # Bottom-up layers
        # Returns a list of the last layers of each stage, 5 in total
        # Don't create the thead (stage 5),so we pick the 4th item in the list
        if callable(config.BACKBONE):
            _, C2, C3, C4, C5 = config.BACKBONE(input_image, stage5=True, train_bn=config.TRAIN_BN)
        else:
            _, C2, C3, C4, C5 = resnet_graph(input_image, config.BACKBONE, stage5=True, train_bn=config.TRAIN_BN)

        # FPN(feature pyramid network) 是接在backbone后面
        # Top-down Layer
        # TODO: add assert to varify feature map sizes match what's in config
        # Up把feature map上采样回来尺寸，1*1是为了减少卷积核个数，减少feature map的个数但是不改变feature map的尺寸（up变尺寸，1*1变维度）
        P5 = KL.Conv2D(config.TOP_DOEN_PYRAMID_SIZE, (1, 1), name='fpn_c5p5')(C5)
        P4 = KL.Add(name='fpn_p4add')([KL.UpSampling2D(size=(2, 2), name='fpn_p5upsampled')(P5),
                                       KL.Conv2D(config.TOP_DOEN_PYRAMID_SIZE, (1, 1), name='fpn_c4p4')(C4)])
        P3 = KL.Add(name='fpn_p3add')([KL.UpSampling2D(size=(2, 2), name='fpn_p4upsampled')(P4),
                                       KL.Conv2D(config.TOP_DOEN_PYRAMID_SIZE, (1, 1), name='fpn_c3p3')(C3)])
        P2 = KL.Add(name='fpn_p2add')([KL.UpSampling2D(size=(2, 2), name='fpn_p3upsampled')(P3),
                                       KL.Conv2D(config.TOP_DOEN_PYRAMID_SIZE, (1, 1), name='fpn_c2p2')(C2)])

        # Attach 3*3 conv to all P layers to get the final feature maps
        P2 = KL.Conv2D(config.TOP_DOEN_PYRAMID_SIZE, (3, 3), padding='SAME', name='fpn_p2')(P2)
        P3 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding='SAME', name='fpn_p3')(P3)
        P4 = KL.Conv2D(config.TOP_DOEN_PYRAMID_SIZE, (3, 3), padding='SAME', name='fpn_p4')(P4)
        P5 = KL.Conv2D(config.TOP_DOEN_PYRAMID_SIZE, (3, 3), padding='SAME', name='fpn_p5')(P5)
        # P6 is used for the 5th anchor scale in RPN. Generated by subsampling from P5 with stride of 2
        P6 = KL.MaxPooling2D(pool_size=(1, 1), strides=2, name='fpn_p6')(P5)

        # Note that P6 is used in rpn,but not in the classifier heads
        rpn_feature_maps = [P2, P3, P4, P5, P6]
        mrcnn_feature_maps = [P2, P3, P4, P5]

        # Anchors
        if mode == 'training':
            anchors = self.get_anchors(config.IMAGE_SHAPE)
            # Duplicates across the batch dimension bacause keras requires it
            # TODO: can this be optimized to avoid duplicationg the anchors?
            anchors = np.broadcast_to(anchors, (config.BATCH_SIZE,) + anchors.shape)
            # A hack to get around Keras's bad support for constants
            anchors = KL.Lambda(lambda x: tf.Variable(anchors), name='anchors')(input_image)
        else:
            # 测试时没有anchor
            anchors = input_anchors

        # RPN model
        rpn = build_rpn_model(config.RPN_ANCHOR_STRIDE, len(config.RPN_ANCHOR_RATIOS), config.TOP_DOEN_PYRAMID_SIZE)
        # Loop through pyramid layers
        layout_outputs = []
        for p in rpn_feature_maps:
            layout_outputs.append(rpn([p]))
        # Concatenate layer outputs
        # Convert from lists of lists of level outputs to list of lists of outputs across levels
        # [[a1,b1,c1],[a2,b2,c2]] => [[a1,a2],[b1,b2],[c1,c2]]
        output_names = ['rpn_class_logits', 'rpn_class', 'rpn_bbox']
        outputs = list(zip(*layout_outputs))
        outputs = [KL.Concatenate(axis=1, name=n)(list(o)) for o, n in zip(outputs, output_names)]

        rpn_class_logits, rpn_class, rpn_bbox = outputs

        # Generate proposals
        # Proposals are [batch,N,(y1,x1,y2,x2)] in normalized coordinates and zero padded
        proposal_count = config.POST_NMS_ROIS_TRAINING if mode == 'training' else config.POST_NMS_ROIS_INFERENCE
        rpn_rois = ProposalLayer(proposal_count=proposal_count, nms_threshold=config.RPN_NMS_THRESHOLD, name='ROI',
                                 config=config)([rpn_class, rpn_bbox, anchors])

        if mode == 'training':
            # Class ID mask to mark class IDs supported by the dataset the image came from
            active_class_ids = KL.Lambda(lambda x: parse_image_meta_graph(x)['active_class_ids'])(input_image_meta)

            if not config.USE_RPN_ROIS:
                # Ingore predicted ROIs and use ROIs provided as an input
                input_rois = KL.Input(shape=[config.POST_NMS_ROIS_INFERENCE, 4], name='input_roi', dtype=np.int32)
                # Normalize coordinates
                target_rois = KL.Lambda(lambda x: denorm_boxes_graph(x, K.shape(input_image)[1:3]))(input_rois)
            else:
                target_rois = rpn_rois

            # Generate detection targets
            # Subsamples proposals and generates target outputs for training
            # Note that proposal class IDs,gt boxes,and gt_masks are zero padded,Equally,returned rois and targets are zero padded
            rois, target_class_ids, target_bbox, target_mask = DetectionTargetLayer(config, name='proposal_target')(
                [target_rois, input_gt_class_ids, gt_boxes, input_gt_masks])

            # Networks heads
            # TODO: verify that this handles zero padded ROIs
            mrcnn_class_logits, mrcnn_class, mrcnn_bbox = fpn_classifier_graph(rois, mrcnn_feature_maps,
                                                                               input_image_meta, config.POOL_SIZE,
                                                                               config.NUM_CLASSES,
                                                                               train_bn=config.TRAIN_BN,
                                                                               fc_layers_size=config.FPN_CLASSIF_FC_LAYERS_SIZE)

            mrcnn_mask = build_fpn_mask_graph(rois, mrcnn_feature_maps, input_image_meta, config.MASK_POOL_SIZE,
                                              config.NUM_CLASSES, train_bn=config.TRAIN_BN)

            # TODO: clean up (use tf.identity if necessary)
            output_rois = KL.Lambda(lambda x: x * 1, name='output_rois')(rois)

            # Losses
            # rpn_feature_maps = [P2, P3, P4, P5, P6]
            # mrcnn_feature_maps = [P2, P3, P4, P5]
            rpn_class_loss = KL.Lambda(lambda x: rpn_class_loss_graph(*x), name='rpn_class_loss')(
                [input_rpn_match, rpn_class_logits])
            rpn_bbox_loss = KL.Lambda(lambda x: rpn_bbox_loss_graph(config, *x), name='rpn_bbox_loss')(
                [input_rpn_bbox, input_rpn_match, rpn_bbox])
            class_loss = KL.Lambda(lambda x: mrcnn_class_loss_graph(*x), name='mrcnn_class_loss')(
                [target_class_ids, mrcnn_class_logits, active_class_ids])
            bbox_loss = KL.Lambda(lambda x: mrcnn_bbox_loss_graph(*x), name='mrcnn_bbox_loss')(
                [target_bbox, target_class_ids, mrcnn_bbox])
            mask_loss = KL.Lambda(lambda x: mrcnn_mask_loss_graph(*x), name='mrcnn_mask_loss')(
                [target_mask, target_class_ids, mrcnn_mask])

            # Model
            inputs = [input_image, input_image_meta, input_rpn_match, input_rpn_bbox, input_gt_class_ids,
                      input_gt_boxes, input_gt_masks]
            if not config.USE_RPN_ROIS:
                inputs.append(input_rois)
            outputs = [rpn_class_logits, rpn_class, rpn_bbox,
                       mrcnn_class_logits, mrcnn_class, mrcnn_bbox, mrcnn_mask,
                       rpn_rois, output_rois,
                       rpn_class_loss, rpn_bbox_loss, class_loss, bbox_loss, mask_loss]
            model = KM.Model(inputs, outputs, name='mask_rcnn')
        else:
            # Network Heads
            # Proposal classifier and BBox regressor heads
            mrcnn_class_logits, mrcnn_class, mrcnn_bbox = fpn_classifier_graph(rpn_rois, mrcnn_feature_maps,
                                                                               input_image_meta, config.POOL_SIZE,
                                                                               config.NUM_CLASSES,
                                                                               train_bn=config.TRAIN_BN,
                                                                               fc_layers_size=config.FPN_CLASSIF_FC_LAYERS_SIZE)

            # Detections
            # Output is [batch,num_detections,(y1,x1,y2,x2,class_id,score)] in normalized coordinates
            detections = DetectionLayer(config, name='mrcnn_detection')(
                [rpn_rois, mrcnn_class, mrcnn_bbox, input_image_meta])

            # Create masks for detections
            detection_boxes = KL.Lambda(lambda x: x[..., :4])(detections)
            mrcnn_mask = build_fpn_mask_graph(detection_boxes, mrcnn_feature_maps, input_image_meta,
                                              config.MASK_POOL_SIZE, config.NUM_CLASSES, train_bn=config.TRAIN_BN)

            model = KM.Model([input_image, input_image_meta, input_anchors],
                             [detections, mrcnn_class, mrcnn_bbox, mrcnn_mask, rpn_rois, rpn_class, rpn_bbox],
                             name='mask_rcnn')

        # Add multi-GPU support
        if config.GPU_COUNT > 1:
            from MaskRCNN.parallel_model import ParallelModel
            model = ParallelModel(model, config.GPU_COUNT)
        return model

    def find_last(self):
        '''
        Finds the last checkpoint file of the last trained model in the model directory
        :return:
        '''
        # Get directory names.Each directory corresponds to a model
        dir_names = next(os.walk(self.model_dir))[1]
        key = self.config.NAME.lower()
        dir_names = filter(lambda f: f.startwith(key), dir_names)
        dir_names = sorted(dir_names)
        if not dir_names:
            import errno
            raise FileNotFoundError(errno.ENOENT, 'Could not find model directory under {}'.format(self.model_dir))
        # Pick last directory
        dir_name = os.path.join(self.model_dir, dir_names[-1])
        # Find the last checkpoint
        checkpoints = next(os.walk(dir_name))[2]
        checkpoints = filter(lambda f: f.startwith('mask_rcnn'), checkpoints)
        checkpoints = sorted(checkpoints)
        if not checkpoints:
            import errno
            raise FileNotFoundError(errno.ENOENT, 'Could not find weight file in {}'.format(dir_name))
        checkpoint = os.path.join(dir_name, checkpoints[-1])
        return checkpoint

    def load_weights(self, filepath, by_name=False, exclude=None):
        '''
        :param filepath:
        :param by_name:
        :param exclude:
        :return:
        '''
        import h5py
        try:
            from  keras.engine import saving
        except ImportError:
            from keras.engine import topology as saving

        if exclude:
            by_name = True

        if h5py is None:
            raise ImportError("'load weights' requires h5py")
        f = h5py.File(filepath, mode='r')
        if 'layer_names' not in f.attrs and 'model_weights' in f:
            f = f['model_weights']

        # In multi-GPU training,we wrap the model.Get layers of the inner model because they have the weights
        keras_model = self.keras_model
        layers = keras_model.inner_model.layers if hasattr(keras_model, 'inner_model') else keras_model.layers

        # Exclude some layers
        if exclude:
            layers = filter(lambda l: l.name not in exclude, layers)
        if by_name:
            saving.load_weights_from_hdf5_group_by_name(f, layers)
        else:
            saving.load_weights_from_hdf5_group_by_name(f, layers)
        if hasattr(f, 'close'):
            f.close()

        # Update the log directory
        self.set_log_dir(filepath)

    def get_imagenet_weights(self):
        '''
        Download Imagenet trained weights from keras
        :return:
        '''
        from keras.utils.data_utils import get_file
        TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/' \
                                 'releases/download/v0.2/' \
                                 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
        weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                TF_WEIGHTS_PATH_NO_TOP,
                                cache_subdir='models',
                                md5_hash='a268eb855778b3df3c7506639542a6af')
        return weights_path

    def compile(self, learning_rate, momentum):
        '''
        Gets the model ready for training.Adds losses,regularization, and metrics Then calls the keras compile() function
        :param learning_rate:
        :param momentum:
        :return:
        '''
        # Optimizer object
        optimizer = keras.optimizers.SGD(lr=learning_rate, momentum=momentum, clipnorm=self.config.GRADIENT_CLIP_NORM)
        # Add losses
        # First,clear previously set losses to avoid duplication
        self.keras_model._losses = []
        self.keras_model._per_input_losses = {}
        losses_names = [
            'rpn_class_loss', 'rpn_bbox_loss', 'mrcnn_class_loss', 'mrcnn_bbox_loss', 'mrcnn_mask_loss'
        ]
        for name in losses_names:
            layer = self.keras_model.get_layer(name)
            if layer.output in self.keras_model.losses:
                continue
            loss = (tf.reduce_mean(layer.output, keep_dims=True) * self.config.LOSS_WEIGHTS.get(name, 1))
            self.keras_model.add_loss(loss)

        # Add L2 Regularization
        # Skip gamma and beta weights of batch normalization layers
        reg_losses = [keras.regularizers.l2(
            self.config.WEIGHT_DECAY)(w) / tf.cast(tf.size(w), tf.float32) for w in self.keras_model.trainable_weights
                      if
                      'gamma' not in w.name and 'beta' not in w.name]
        self.keras_model.add_loss(tf.add_n(reg_losses))

        # Compile
        self.keras_model.compile(optimizer=optimizer, loss=[None] * len(self.keras_model.outputs))

        # Add metrics for losses
        for name in losses_names:
            if name in self.keras_model.metrics_names:
                continue
            layer = self.keras_model.get_layer(name)
            self.keras_model.metrics_names.append(name)
            loss = (tf.reduce_mean(layer.output, keepdims=True) * self.config.LOSS_WEIGHTS.get(name, 1))
            self.keras_model.metrics_tensors.append(loss)

    def set_trainable(self, layer_regex, keras_model=None, indent=0, verbose=1):
        '''
        sets model layers as trainable if their names match the given regular expression
        :param layer_regex:
        :param keras_model:
        :param indent:
        :param verbose:
        :return:
        '''
        # Print message on the first call (but not on recursive calls)
        if verbose > 0 and keras_model is None:
            log('Selecting layers to train')

        keras_model = keras_model or self.keras_model

        # In multi-GPU training,we wrap the model.Get layers of the inner model because they have the weights
        layers = keras_model.inner_model.layers if hasattr(keras_model, 'inner_model') else keras_model.layers

        for layer in layers:
            if layer.__class__.__name__ == 'Model':
                print('In model:', layer.name)
                self.set_trainable(
                    layer_regex, keras_model=layer, indent=indent + 4
                )
                continue
            if not layer.weights:
                continue
            trainable = bool(re.fullmatch(layer_regex, layer.name))
            # Update layer.If layer is a container,update inner layer
            if layer.__class__.__name__ == 'TimeDistributed':
                layer.layer.trainable = trainable
            else:
                layer.trainable = trainable
            # Print trainable layer names
            if trainable and verbose > 0:
                log('{}{:20} ({})'.format('' * indent, layer.name, layer.__class__.__name__))

    def set_log_dir(self, model_path=None):
        '''
        sets the model log directory and epoch counter
        :param model_path:
        :return:
        '''
        # Set date and epoch counter as if starting a new model
        self.epoch = 0
        now = datetime.datetime.now()

        # If we have a model path with date and epochs use them
        if model_path:
            # Continue from we left of.Get epoch and date from the file name
            # A simple model path might look like:
            # \path\to\logs\coco21171029T2315\mask_rcnn_coco_0001.h5
            regex = r".*[/\\][\w-]+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})[/\\]mask\_rcnn\_[\w-]+(\d{4})\.h5"
            m = re.match(regex, model_path)
            if m:
                now = datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)),
                                        int(m.group(5)))
                # Epoch number in file is 1-based,and in keras code it's 0-based
                # So,adjust for that then increment by one to start from the next epoch
                self.epoch = int(m.group(6)) - 1 + 1
                print('Re-starting from epoch %d' % self.epoch)

        # Directory for training logs
        self.log_dir = os.path.join(self.model_dir, '{} {:%Y%m%dT%H%M}'.format(self.config.NAME.lower(), now))

        # Path to save after each epoch. Include placeholders that get filled by keras
        self.checkpoint_path = os.path.join(self.log_dir, 'mask_rcnn_{}_*epoch*.h5'.format(self.config.NAME.lower()))
        self.checkpoint_path = self.checkpoint_path.replace('*epoch*', '{epoch:04d}')

    def train(self, train_dataset, val_dataset, learning_rate, epochs, layers, augmentation=None, custom_callbacks=None,
              no_augmentation_sources=None):
        '''
        Train the model
        :param train_dataset: Training and validation Dataset objects
        :param val_dataset:
        :param learning_rate:
        :param epochs: Number of training epochs. Note the previous training epochs are considered to be done already, so this
        actually determines the epochs to train in total rather than in this actually determines the epochs to train in total rather than in this particaular call.
        :param layers: Allows selecting which layer names to train
            A regular expression to match layer names to train
            One of these predefined values
            heads: The RPN,classifier and masks heads of the network
            all: All the layers
            3+: Train Resnet stage 3 and up
            4+: Train resnet stage 4 and up
            5+: Train resnet stage 5 and up
        :param augmentation: Optional. Imgaug
        :param custom_callback: Optional. Add custom callbacks to be called with the keras fit_generator method.
        :param no_augmentation_sources:
        :return:
        '''
        assert self.mode == 'training', 'Create model in training mode'

        # Pre-defined layer regular expressions
        layer_regeex = {
            # all layers but the backbone
            'heads': r'mrcnn\_.*|(rpn\_.*)|(fpn\_.*)',
            # From a specific resnet stage and up
            '3+': r'(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)',
            '4+': r'(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)',
            '5+': r'(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)',
            # All layers
            'all': '.*',
        }
        if layers in layer_regeex.keys():
            layers = layer_regeex[layers]

        # Data generators
        train_generator = data_generator(train_dataset, self.config, shuffle=True, augmentation=augmentation,
                                         batch_size=self.config.BATCH_SIZE,
                                         no_augmentation_sources=no_augmentation_sources)
        val_generator = data_generator(val_dataset, self.config, shuffle=True, batch_size=self.config.BATCH_SIZE)

        # Create log_dir if it does not exist:
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # Callbacks
        callbacks = [
            keras.callbacks.TensorBoard(log_dir=self.log_dir, histogram_freq=0, write_graph=True, write_images=False),
            keras.callbacks.ModelCheckpoint(self.checkpoint_path, verbose=0, save_weights_only=True),
        ]

        # Add custom callbacks to the list
        if custom_callbacks:
            callbacks += custom_callbacks

        # Train
        log('\n Starting at epoch {}.LR={}\n'.format(self.epoch, learning_rate))
        log('Checkpoint path:{}'.format(self.checkpoint_path))
        self.set_trainable(layers)
        self.compile(learning_rate, self.config.LEARNING_RATE)

        # Work-around for windows: keras fails on windows when use multiprocessing works.see discussion here:
        if os.name is 'nt':
            workers = 0
        else:
            workers = multiprocessing.cpu_count()

        self.keras_model.fit_generator(train_generator, initial_epoch=self.epoch, epochs=epochs,
                                       steps_per_eppoch=self.config.STEPS_PER_EPOCH, callbacks=callbacks,
                                       validation_data=val_generator, validation_steps=self.config.VALIDATION_STEPS,
                                       max_queue_size=100, workers=workers, use_multiprocessing=True)
        self.epoch = max(self.epoch, epochs)

    def mold_inputs(self, images):
        '''
        Takes a list of images and modifies them to the format expected as an input to the neural network
        :param images:
        :return:
        '''
        molded_images = []
        image_metas = []
        windows = []
        for image in images:
            # Resize image
            # TODO: move resizing to mold_image()
            molded_image, window, scale, padding, crop = utils.resize_image(image, min_dim=self.config.IMAGE_MAX_DIM,
                                                                            min_scale=self.config.IMAGE_MIN_SCALE,
                                                                            max_dim=self.config.IMAGE_MAX_DIM,
                                                                            mode=self.config.IMAGE_RESIZE_MODE)
            molded_image = mold_image(molded_image, self.config)
            # Build image_meta
            image_meta = compose_image_meta(0, image.shape, molded_image.shape, window, scale,
                                            np.zeros([self.config.NUM_CLASSES], dtype=np.int32))
            # Append
            molded_images.append(molded_image)
            windows.append(window)
            image_metas.append(image_meta)
        # Pack into arrays
        molded_images = np.stack(molded_images)
        image_metas = np.stack(image_metas)
        windows = np.stack(windows)
        return molded_images, image_metas, windows

    def unmold_detections(self,detections,mrcnn_mask,original_image_shape,image_shape,window):
        '''
        Reformats the detections of one image from the format of the neural network output to a format suitable for use in the rest of the application
        :param detections:
        :param mrcnn_mask:
        :param original_image_shape:
        :param image_shape:
        :param window:
        :return:
        '''







    def get_anchors(self, image_shape):
        '''
        returns anchor pyramidfor the given image size
        :param image_shape:
        :return:
        '''
        backbone_shape = compute_backbone_shapes(self.config, image_shape)
        # Cache anchors and reuse if image shape is the same
        if not hasattr(self, '_anchor_cache'):
            self._anchor_cache = {}
        if not tuple(image_shape) in self._anchor_cache:
            # Generate Anchors
            a = utils.generate_pyramid_anchors(self.config.RPN_ANCHOR_SCALES, self.config.RPN_ANCHOR_RATIOS,
                                               backbone_shape,
                                               self.config.BACKBONE_STRIDES, self.config.RPN_ANCHOR_STRIDE)
            # Keep a copy of the latest anchors in pixel coordinates because it's used in the inspect_model notebooks
            # TODO: Remove this after the notebook are refactored to not use it
            self.anchors = a
            # Normalize coordinates
            self._anchor_cache[tuple(image_shape)] = utils.norm_boxes(a, image_shape[:2])
        return self._anchor_cache[tuple(image_shape)]


# Data Formatting
def compose_image_meta(image_id, orignal_image_shape, image_shape, window, scale, active_class_ids):
    pass


def parse_image_meta(meta):
    pass


def parse_image_meta_graph(meta):
    '''
    Parses a tensor that contains image attributes to its components
    把每一部分都取出来了，给了一个字典形式
    :param meta: [batch,meta length] where meta length depends on num_classes
    :return:
    '''
    # TODO: print meta.shape
    image_id = meta[:, 0]
    original_image_shape = meta[:, 1:4]
    image_shape = meta[:, 4:7]
    window = meta[:, 7:11]  # (y1,x1,y2,x2) window of image in pixels
    scale = meta[:, 11]
    active_class_ids = meta[:, 12:]
    return {
        'image_id': image_id,
        'original_image_shape': original_image_shape,
        'image_shape': image_shape,
        'window': window,
        'scale': scale,
        'active_class_ids': active_class_ids,
    }


def mold_image(images, config):
    pass


def unmold_image(normalized_images, config):
    pass


# Miscellenous Graph function
def trim_zeros_graph(boxes, name='trim_zeros'):
    '''
    boxes应该是四个坐标的矩阵，但是里面也有0的，要把0给去掉
    :param boxes: [N,4] matrix of boxes
    :param name:  [N] a 1D boolean mask identifying the rows to keep
    :return:
    '''
    non_zeros = tf.cast(tf.reduce_sum(tf.abs(boxes), axis=1), tf.bool)
    boxes = tf.boolean_mask(boxes, non_zeros, name=name)
    return boxes, non_zeros


def batch_pack_graph(x, counts, num_rows):
    '''
    picks different number of values from each row in x depending on the values in counts
    :param x:
    :param counts:
    :param num_rows:
    :return:
    '''
    outputs = []
    for i in range(num_rows):
        outputs.append(x[i, :counts[i]])
    return tf.concat(outputs, axis=0)


def norm_boxes_fraph(boxes, shape):
    '''
    Converts boxes from pixel coordinates to normalized coordinates
    :param boxes: [...,(y1,x1,y2,x2)] in pixel coordinates
    :param shape: [...,(height,width)] in pixels
    Notes: In pixel coordinates (y2,x2) is outside the box,but in normalized coordinates it's inside the box
    :return:
      [...,(y1,x1,y2,x2)] in normalized coordinates
    '''
    h, w = tf.split(tf.cast(shape, tf.float32), 2)
    scale = tf.concat([h, w, h, w], axis=-1) - tf.constant(1.0)
    shift = tf.constant([0, 0, 1, 1])
    return tf.divide(boxes - shift, scale)


def denorm_boxes_graph(boxes, shape):
    '''
    Converts boxes from normalized coordinates to pixel coordinates
    从归一化里面出来变成正常的尺寸
    :param boxes:  [,(y1,x1,y2,x2)] in pixel coordinates
    :param shape:
    :return:
    '''
    h, w = tf.split(tf.cast(shape, tf.float32), 2)
    scale = tf.concat([h, w, h, w], axis=-1) - tf.constant(1.0)
    shift = tf.constant([0., 0., 1., 1.])
    return tf.cast(tf.round(tf.multiply(boxes, scale) + shift), tf.int32)
