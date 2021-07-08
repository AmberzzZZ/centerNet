import keras.backend as K
import tensorflow as tf


def kp_loss(args, n_classes, pos_thresh=1.0):
    # heatmaps: [b,h,w,c]
    # offsets: [b,h,w,2]
    # gt: [b,h,w,c+2]
    heatmaps, offsets, gt = args
    pos_mask = tf.where(gt[...,:n_classes]>=1, tf.ones_like(gt[...,:n_classes]), tf.zeros_like(gt[...,:n_classes]))
    # pos_mask = tf.where(gt[...,:n_classes]>=pos_thresh, gt[...,:n_classes], tf.zeros_like(gt[...,:n_classes]))    # soft
    pos_mask_all = K.sum(pos_mask, axis=-1, keepdims=True)
    # heatmap loss: focal loss with penalty reduction, on positives & negatives
    loss_kp = focal_loss(gt[...,:n_classes], heatmaps, pos_thresh)
    # offsets loss: L1 loss using raw pixels
    loss_offset = l1_loss(gt[...,n_classes:n_classes+2], offsets) * pos_mask_all
    # norm term
    pos_cnt = K.maximum(1., K.sum(pos_mask, axis=[1,2,3]))
    loss_kp = K.sum(loss_kp, axis=[1,2,3]) / pos_cnt
    loss_kp = K.mean(loss_kp)
    loss_offset = K.sum(loss_offset, axis=[1,2,3]) / pos_cnt
    loss_offset = K.mean(loss_offset)
    # sum
    loss = loss_kp + loss_offset
    loss = tf.Print(loss, [loss_kp, loss_offset], message=" loss kp & loss offsets: ")

    return loss


def det_loss(args, n_classes, pos_thresh=1.0):
    # heatmaps: [b,h,w,c]
    # offsets: [b,h,w,2]
    # sizes: [b,h,w,2]
    # gt: [b,h,w,c+2+2]
    heatmaps, offsets, sizes, gt = args
    pos_mask = tf.where(gt[...,:n_classes]>=1, tf.ones_like(gt[...,:n_classes]), tf.zeros_like(gt[...,:n_classes]))
    # pos_mask = tf.where(gt[...,:n_classes]>=pos_thresh, gt[...,:n_classes], tf.zeros_like(gt[...,:n_classes]))   # soft
    pos_mask_all = K.sum(pos_mask, axis=-1, keepdims=True)
    # heatmap loss: focal loss with penalty reduction, on positives & negatives
    loss_kp = focal_loss(gt[...,:n_classes], heatmaps, pos_thresh)
    # offsets loss: L1 loss using raw pixels, on positives
    loss_offset = l1_loss(gt[...,n_classes:n_classes+2], offsets) * pos_mask_all
    # sizes loss: L1 loss using raw pixels, on positives
    loss_size = l1_loss(gt[...,n_classes+2:n_classes+4], sizes) * pos_mask_all
    # norm term
    pos_cnt = K.maximum(1., K.sum(pos_mask, axis=[1,2,3]))
    loss_kp = K.sum(loss_kp, axis=[1,2,3]) / pos_cnt
    loss_kp = K.mean(loss_kp)
    loss_offset = K.sum(loss_offset, axis=[1,2,3]) / pos_cnt
    loss_offset = K.mean(loss_offset)
    loss_size = K.sum(loss_size, axis=[1,2,3]) / pos_cnt
    loss_size = K.mean(loss_size)
    # sum
    loss = loss_kp + loss_offset + loss_size*0.1
    loss = tf.Print(loss, [loss_kp, loss_offset, loss_size], message=" loss kp & loss offsets & loss sizes: ")

    return loss


def inter_loss(args, n_classes, pos_thresh=0.7):
    # [inter heatmaps]: [b,h,w,c]
    # gt: [b,h,w,c+2]
    inter_heatmaps = args[:-1]
    gt = args[-1]
    inter_loss = 0.
    for idx, heatmaps in enumerate(inter_heatmaps):
        pos_mask = tf.where(gt[...,:n_classes]>=pos_thresh, gt[...,:n_classes], tf.zeros_like(gt[...,:n_classes]))
        # heatmap loss: focal loss with penalty reduction, on positives & negatives
        loss_kp = focal_loss(gt[...,:n_classes], heatmaps, pos_thresh)
        # norm term
        pos_cnt = K.maximum(1., K.sum(pos_mask, axis=[1,2,3]))
        loss_kp = K.sum(loss_kp, axis=[1,2,3]) / pos_cnt
        loss_kp = K.mean(loss_kp)
        # sum
        inter_loss += loss_kp
        inter_loss = tf.Print(inter_loss, [idx, loss_kp], message="stage & loss kp : ")

    return inter_loss


def focal_loss(y_true, y_pred, pos_thresh, alpha=2, beta=4):
    pt = 1 - K.abs(y_true - y_pred)
    pt = K.clip(pt, K.epsilon(), 1-K.epsilon())
    focal_loss_ = - tf.pow(1-pt, alpha) * tf.log(pt)
    # penalty reduction on negatives, reweighting on postives
    focal_loss_ = tf.where(y_true<pos_thresh,
                           tf.pow(pos_thresh-y_true+1., beta)*focal_loss_,
                           tf.pow(y_true-pos_thresh+1., 10)*focal_loss_)
    return focal_loss_


def l1_loss(y_true, y_pred):
    l1_loss_ = K.abs(y_true - y_pred)
    return l1_loss_










