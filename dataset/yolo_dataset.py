import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
from .augmentations import augment_image, mosaic_augment, mixup_augment
from config import IMG_SIZE, MOSAIC, MIXUP

class YoloDataset(Dataset):
    def __init__(self, images_dir, labels_dir, img_size=IMG_SIZE):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.img_size = img_size

        self.images = sorted([os.path.join(images_dir,f) for f in os.listdir(images_dir) if f.endswith(('.jpg','.png'))])
        self.labels = sorted([os.path.join(labels_dir,f) for f in os.listdir(labels_dir) if f.endswith('.txt')])
        assert len(self.images) == len(self.labels)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label_path = self.labels[idx]
        img = Image.open(img_path).convert('RGB')
        img = np.array(img.resize((self.img_size,self.img_size)))

        boxes = []
        with open(label_path,'r') as f:
            for line in f.readlines():
                cls, x, y, w, h = map(float, line.strip().split())
                boxes.append([cls,x,y,w,h])
        boxes = np.array(boxes) if boxes else np.zeros((0,5))

        if MOSAIC and np.random.rand()<0.5:
            img, boxes = mosaic_augment(img, boxes, self.images, self.labels)
        if MIXUP and np.random.rand()<0.5:
            img, boxes = mixup_augment(img, boxes, self.images, self.labels)
        img, boxes = augment_image(img, boxes)

        img = torch.tensor(img.transpose(2,0,1)/255.0,dtype=torch.float32)
        boxes = torch.tensor(boxes,dtype=torch.float32)

        return img, boxes
