import torch

def xywh2xyxy(boxes):
    x,y,w,h = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    x1 = x-w/2
    y1 = y-h/2
    x2 = x+w/2
    y2 = y+h/2
    return torch.stack([x1,y1,x2,y2],dim=1)

def bbox_iou(box1,box2,eps=1e-7):
    inter_x1 = torch.max(box1[:,0],box2[0])
    inter_y1 = torch.max(box1[:,1],box2[1])
    inter_x2 = torch.min(box1[:,2],box2[2])
    inter_y2 = torch.min(box1[:,3],box2[3])
    inter_area = (inter_x2-inter_x1).clamp(0)*(inter_y2-inter_y1).clamp(0)
    area1 = (box1[:,2]-box1[:,0])*(box1[:,3]-box1[:,1])
    area2 = (box2[2]-box2[0])*(box2[3]-box2[1])
    iou = inter_area/(area1+area2-inter_area+eps)
    return iou

def ciou_loss(pred_boxes,target_boxes):
    iou = bbox_iou(pred_boxes,target_boxes)
    px = (pred_boxes[:,0]+pred_boxes[:,2])/2
    py = (pred_boxes[:,1]+pred_boxes[:,3])/2
    tx = (target_boxes[0]+target_boxes[2])/2
    ty = (target_boxes[1]+target_boxes[3])/2
    rho2 = (px-tx)**2 + (py-ty)**2
    c2 = (max(pred_boxes[:,2].max(),target_boxes[2])-min(pred_boxes[:,0].min(),target_boxes[0]))**2 + \
         (max(pred_boxes[:,3].max(),target_boxes[3])-min(pred_boxes[:,1].min(),target_boxes[1]))**2
    ciou = iou - rho2/c2
    return 1-iou
