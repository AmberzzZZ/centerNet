import keras.backend as K
import tensorflow as tf


def kp_loss(args, n_classes):
    # heatmaps: [b,h,w,c]
    # offsets: [b,h,w,2]
    # gt: [b,h,w,c+2]
    heatmaps, offsets, gt = args
    pos_mask = tf.where(gt[...,:n_classes]>=1, tf.ones_like(gt[...,:n_classes]), tf.zeros_like(gt[...,:n_classes]))
    pos_mask_all = K.sum(pos_mask, axis=-1, keepdims=True)
    # heatmap loss: focal loss with penalty reduction, on positives & negatives
    loss_kp = focal_loss(gt[...,:n_classes], heatmaps)
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


def det_loss(args, n_classes):
    # heatmaps: [b,h,w,c]
    # offsets: [b,h,w,2]
    # sizes: [b,h,w,2]
    # gt: [b,h,w,c+2+2]
    heatmaps, offsets, sizes, gt = args
    pos_mask = tf.where(gt[...,:n_classes]>=1, tf.ones_like(gt[...,:n_classes]), tf.zeros_like(gt[...,:n_classes]))
    pos_mask_all = K.sum(pos_mask, axis=-1, keepdims=True)
    # heatmap loss: focal loss with penalty reduction, on positives & negatives
    loss_kp = focal_loss(gt[...,:n_classes], heatmaps)
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


def focal_loss(y_true, y_pred, alpha=2, beta=4):
    pt = 1 - K.abs(y_true - y_pred)
    pt = K.clip(pt, K.epsilon(), 1-K.epsilon())
    focal_loss_ = - tf.pow(1-pt, alpha) * tf.log(pt)
    # penalty reduction on negatives, reweighting on postives
    focal_loss_ = tf.where(y_true<1, tf.pow(1-y_true, beta)*focal_loss_, 100*focal_loss_)
    return focal_loss_


def l1_loss(y_true, y_pred):
    l1_loss_ = K.abs(y_true - y_pred)
    return l1_loss_











