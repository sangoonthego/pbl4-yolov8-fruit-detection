import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from utils.bbox import xywh2xyxy, ciou_loss

bce = nn.BCEWithLogitsLoss()

def yolo_loss(preds, targets, anchors, num_classes):
    # preds: [B,C,H,W], targets: [B,N,5]
    total_loss=0
    B = preds.shape[0]
    for i in range(B):
        t = targets[i]
        if t.shape[0]==0: continue
        # split prediction
        pred_boxes = preds[i].view(-1,4)
        pred_obj = preds[i].view(-1,1)[:,0]
        pred_cls = preds[i].view(-1,num_classes)
        target_boxes = xywh2xyxy(t[:,1:5])
        ciou = ciou_loss(pred_boxes,target_boxes).mean()
        obj_loss = bce(pred_obj,torch.ones_like(pred_obj))
        cls_loss = bce(pred_cls,nn.functional.one_hot(t[:,0].long(),num_classes).float())
        total_loss += ciou + obj_loss + cls_loss.mean()
    return total_loss
