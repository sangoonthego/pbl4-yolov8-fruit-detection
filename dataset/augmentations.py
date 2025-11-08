import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import numpy as np
import random
from PIL import Image

from config import IMG_SIZE

def augment_image(img, boxes):
    # Horizontal flip
    if random.random()<0.5:
        img = cv2.flip(img,1)
        if boxes.shape[0]>0:
            boxes[:,1] = 1-boxes[:,1]  # x_center normalized
    # Random color jitter
    if random.random()<0.5:
        alpha = 1+0.2*(random.random()-0.5)
        beta = 10*(random.random()-0.5)
        img = np.clip(img*alpha+beta,0,255).astype(np.uint8)
    return img, boxes

def mosaic_augment(img, boxes, images, labels):
    # Simple 4-image mosaic
    h, w = img.shape[:2]
    s = h
    xc, yc = int(s/2), int(s/2)
    new_img = np.zeros((s,s,3),dtype=np.uint8)
    new_boxes = []

    indices = np.random.randint(0,len(images),3)
    imgs = [img]+[np.array(Image.open(images[i]).resize((IMG_SIZE,IMG_SIZE))) for i in indices]
    boxes_list = [boxes]+[np.loadtxt(labels[i]).reshape(-1,5) if os.path.exists(labels[i]) else np.zeros((0,5)) for i in indices]

    for i, (im, b) in enumerate(zip(imgs, boxes_list)):
        x_offset = 0 if i%2==0 else xc
        y_offset = 0 if i<2 else yc
        h_im, w_im = im.shape[:2]
        new_img[y_offset:y_offset+h_im,x_offset:x_offset+w_im] = im
        if b.shape[0]>0:
            b_copy = b.copy()
            b_copy[:,1] = (b_copy[:,1]*w_im+x_offset)/s
            b_copy[:,2] = (b_copy[:,2]*h_im+y_offset)/s
            b_copy[:,3] = b_copy[:,3]*w_im/s
            b_copy[:,4] = b_copy[:,4]*h_im/s
            new_boxes.append(b_copy)
    if new_boxes:
        new_boxes = np.concatenate(new_boxes,axis=0)
    else:
        new_boxes = np.zeros((0,5))
    return new_img,new_boxes

def mixup_augment(img, boxes, images, labels):
    idx = np.random.randint(0,len(images))
    img2 = np.array(Image.open(images[idx]).resize((IMG_SIZE,IMG_SIZE)))
    if os.path.exists(labels[idx]):
        boxes2 = np.loadtxt(labels[idx]).reshape(-1,5)
    else:
        boxes2 = np.zeros((0,5))
    alpha = 0.5
    new_img = (img*alpha+img2*(1-alpha)).astype(np.uint8)
    new_boxes = np.vstack([boxes, boxes2]) if boxes2.shape[0]>0 else boxes
    return new_img,new_boxes
