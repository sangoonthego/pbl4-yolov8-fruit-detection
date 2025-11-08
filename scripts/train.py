import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.utils.data import DataLoader
from dataset.yolo_dataset import YoloDataset
from model.yolo import YOLOv8
from utils.anchors import assign_anchors
from loss import yolo_loss
from config import *
import os

train_ds = YoloDataset(TRAIN_IMAGES,TRAIN_LABELS)
train_loader = DataLoader(train_ds,batch_size=BATCH_SIZE,shuffle=True)
val_ds = YoloDataset(VAL_IMAGES,VAL_LABELS)
val_loader = DataLoader(val_ds,batch_size=BATCH_SIZE,shuffle=False)

model = YOLOv8(NUM_CLASSES).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(),lr=LR)
os.makedirs(SAVE_DIR,exist_ok=True)

anchors = torch.tensor([[10,13],[16,30],[33,23]])/IMG_SIZE

for epoch in range(EPOCHS):
    model.train()
    total_loss=0
    for imgs, targets in train_loader:
        imgs = imgs.to(DEVICE)
        preds = model(imgs)
        loss = yolo_loss(preds, targets, anchors, NUM_CLASSES)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss:{total_loss/len(train_loader):.4f}")
    if (epoch+1)%5==0:
        torch.save(model.state_dict(), os.path.join(SAVE_DIR,f'yolov8_epoch{epoch+1}.pt'))
