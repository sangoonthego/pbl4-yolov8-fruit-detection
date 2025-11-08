# utils/nms.py
import torch
from .bbox import bbox_iou

def non_max_suppression(boxes, conf, iou_thresh=0.45):
    keep=[]
    idxs = conf.argsort(descending=True)
    while idxs.numel()>0:
        i = idxs[0]
        keep.append(i)
        if idxs.numel()==1: break
        ious = bbox_iou(boxes[i:i+1], boxes[idxs[1:]])
        idxs = idxs[1:][ious<iou_thresh]
    return keep
