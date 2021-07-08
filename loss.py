import keras.backend as K
import tensorflow as tf


def centerNet_loss(args, levels, n_classes):
    # cls_heads: cls logits, [0,1]
    # box_heads: reg values, xcyc2orgin, [0, 1], wh2featuremap, unlimited pos
    # cls_agn_heads: centerness regression values, regress in [-1,0,1]
    cls_heads = args[:levels]
    box_heads = args[levels:2*levels]
    cls_agn_heads = args[2*levels:3*levels]
    gts = args[3*levels:4*levels]
    s1_factor = 0.5
    radius_factor = 0.1
    loss = 0.
    # loss per level
    for cls_head, box_head, cls_agn_head, gt in zip(cls_heads, box_heads, cls_agn_heads, gts):

        # raw outputs to logits / regressions : sigmoid / exp
        cls_head = K.sigmoid(cls_head)     # hard
        box_head = K.exp(box_head)
        cls_agn_head = K.sigmoid(cls_agn_head)     # soft

        #### centerness loss: l1 loss
        loss_centerness = l1_loss(cls_agn_head, gt[...,n_classes:n_classes+1])   # [b,h,w,1]
        valid_mask = tf.where(cls_agn_head>.5, tf.ones_like(cls_agn_head), tf.zeros_like(cls_agn_head))  # fg spatial map

        #### cls loss: bce loss
        loss_cls = bce_loss(cls_head, gt[..., :n_classes], valid_mask)

        #### reg loss: l1 loss
        loss_reg_center = l1_loss(box_head[...,-4:-2], gt[...,-4:-2], valid_mask)
        loss_reg_radius = l1_loss(box_head[...,-2:], gt[...,-2:], valid_mask)
        loss_reg = loss_reg_center + radius_factor*loss_reg_radius

        loss += s1_factor*loss_centerness + loss_cls + loss_reg
        loss = tf.Print(loss, [loss_centerness, loss_cls, loss_reg], message='centerness & cls & reg per level')

    loss = tf.Print(loss, [loss], message='total loss')
    return loss


def l1_loss(pred, gt, valid_mask=None):
    if valid_mask is None:
        valid_mask = tf.ones_like(pred)

    loss = K.sum(K.abs(pred-gt)*valid_mask) / K.sum(valid_mask)
    return loss


def bce_loss(pred, gt, valid_mask):
    if valid_mask is None:
        valid_mask = tf.ones_like(pred)

    pt_1 = tf.where(gt>.5, pred, tf.ones_like(pred))
    pt_0 = tf.where(gt<.5, pred, tf.zeros_like(pred))
    bce = -K.log(pt_1) - K.log(1-pt_0)
    loss = K.sum(bce*valid_mask) / K.sum(valid_mask)
    return loss




