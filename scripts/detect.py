import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import torch
from model.yolo import YOLOv8
from utils.nms import non_max_suppression
from config import *

def detect_image(model, img_path):
    img = cv2.imread(img_path)
    h0,w0 = img.shape[:2]
    img_resized = cv2.resize(img,(IMG_SIZE,IMG_SIZE))
    img_input = torch.tensor(img_resized.transpose(2,0,1)/255.0,dtype=torch.float32).unsqueeze(0).to(DEVICE)

    model.eval()
    with torch.no_grad():
        pred = model(img_input)
    # dummy detection
    boxes = torch.rand((10,4))*IMG_SIZE
    conf = torch.rand(10)
    keep = non_max_suppression(boxes,conf,NMS_THRESH)

    for i in keep:
        x1,y1,x2,y2 = map(int,boxes[i])
        cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
    cv2.imshow("Detect",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
