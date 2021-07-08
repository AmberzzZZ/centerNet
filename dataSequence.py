from keras.utils import Sequence
import cv2
import numpy as np
import os
import pandas as pd
import math
import random
import json
from aug import resize_with_box, crop_with_box, flip_with_box


label_dict = {}


class dataSequence(Sequence):

    def __init__(self, img_dir, anno_dir, batch_size, input_shape, n_classes, shuffle=True):
        super(dataSequence, self).__init__()
        self.img_dir = img_dir
        self.anno_dir = anno_dir
        self.batch_size = batch_size
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.shuffle = shuffle

    def __len__(self):
        return math.ceil(len(self.full_lst)) / float(self.batch_size)

    def __getitem__(self, index):
        batch_indices = self.indices[self.batch_size*index: self.batch_size*(index+1)]
        batch_data = [self.full_lst[k] for k in batch_indices]
        x_batch, y_batch = self.data_generator(batch_data)
        return x_batch, y_batch

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def data_generator(self, batch_lst):
        h, w = self.input_shape
        b = self.batch_size
        c = self.n_classes
        img_batch = np.zeros((b, h, w, 3))
        gt_batch = [np.zeros((b,h//s,w//s,c+1+4)) for s in [8,16,32,64,128]]

        for i, file_name in enumerate(batch_lst):
            try:
                img = cv2.imread(os.path.join(self.img_dir, file_name), 1)
                boxes = read_json(os.path.join(self.anno_dir, file_name+'.json'))   # [N,5]
                # aug
                img, boxes = det_aug(img, boxes, self.input_shape)
                img_batch[i] = img
                gt_batch[i] = prep_gt(boxes, h, w, c)
            except:
                print("wrong img / label file %s" % file_name)
                img = np.zeros((h,w,3))
                img_batch[i] = img
            return [img_batch, *gt_batch], np.zeros((self.batch_size))


def read_json(yolo_json_file):
    # get box from json: [M,5], x1y1x2y2c
    f = open(yolo_json_file, 'r')
    boxes = json.loads(f.read())
    box_arr = []
    for b in boxes:
        if b['label'] not in label_dict.keys():
            continue
        if min([b['x1'],b['x2'],b['y1'],b['y2']])>0 and max([b['x1'],b['x2'],b['y1'],b['y2']])<1:
            cls_id = label_dict[b['label']]
            xc = (b['x1']+b['x2']) / 2.
            yc = (b['y1']+b['y2']) / 2.
            w = b['x2'] - b['x1']
            h = b['y2'] - b['y1']
            if min([w,h])>0 and max([w,h])<1:
                box_arr.append([xc,yc,w,h,cls_id])
    f.close()
    return np.array(box_arr)


def det_aug(img, boxes, input_shape):
    # strong aug: efficient det aug, randAug
    # easy aug: FRCNN aug, flip & crop & colorjit, fixed shot edge: 800
    # step1: resize by shorter side 768
    short = input_shape[0]
    h, w, _ = img.shape
    if h>w:
        hh = short*h/w
        img, boxes = resize_with_box(img, boxes, (short,hh))   # target wh
    else:
        ww = short*w/h
        img, boxes = resize_with_box(img, boxes, (ww,short))   # target wh
    # random crop to [short,short]
    if hh:
        lbound = random.randint(0, hh-short)
        img, boxes = crop_with_box(img, boxes, lbound, vertical=True)
    else:
        lbound = random.randint(0, ww-short)
        img, boxes = crop_with_box(img, boxes, lbound, vertical=False)
    # random flip
    if random.uniform(0, 1)>0.5:
        img, boxes = flip_with_box(img, boxes)
    return img, boxes


def prep_gt(boxes, img_h, img_w, n_classes, pos_thresh=0.9, neg_thresh=0.5):
    # use box to generate heatmap per cls & reg values
    soi = [[0, 80], [64, 160], [128, 320], [256, 640], [512, 10000000]]
    strides = [8,16,32,64,128]
    gt = [np.zeros((h//s,w//s,n_classes+1+4)) for s in strides]
    for b in boxes:
        xc,yc,w,h,cls_id = b       # normed values
        xc,yc,w,h = xc*img_w, yc*img_h, w*img_w, h*img_h    # abs values
        max_edge = max(w,h)
        # assign levels by scale
        for i, [lbound,hbound] in enumerate(soi):
            if lbound<=max_edge<=hbound:
                # reg values
                grid_x, grid_y = xc//strides[i], yc//strides[i]
                reg_x, reg_y = xc/strides[i]-grid_x, xc/strides[i]-grid_y
                reg_w, reg_h = w/strides[i], h/strides[i]
                # cls(logits) & centerness(reg)
                coords_x, coords_y = np.meshgrid(np.arange(w//strides[i]), np.arange(h//strides[i]))  # [hs,ws,1]
                gt[:,:,c][grid_y-h//2:grid_y+h//2, grid_x-w//2:grid_x+w//2] = 1
                distance = np.sqrt((coords_x-grid_x)**2 + (coords_y-grid_y)**2)   # [hs,ws,1]
                centerness = np.exp(-distance)
                gt[:,:,n_classes] = -1
                gt[:,:,n_classes][centerness>pos_thresh] = centerness[centerness>pos_thresh]
                gt[:,:,n_classes][centerness>neg_thresh] = centerness[centerness<neg_thresh]
                gt[grid_y,grid_x,n_classes+1:] = [reg_x,reg_y,reg_w,reg_h]
    return gt






        