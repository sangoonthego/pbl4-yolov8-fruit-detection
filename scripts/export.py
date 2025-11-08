import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from model.yolo import YOLOv8
from config import NUM_CLASSES, SAVE_DIR

def export_model(checkpoint_path, export_path='yolov8_export.pt'):
    model = YOLOv8(NUM_CLASSES)
    model.load_state_dict(torch.load(checkpoint_path,map_location='cpu'))
    torch.save(model.state_dict(),export_path)
    print(f"Exported model to {export_path}")
