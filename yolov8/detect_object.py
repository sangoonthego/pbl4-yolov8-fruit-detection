import os
import sys
import cv2
from pathlib import Path
from ultralytics import YOLO
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

model_path = "runs/detect/train1/weights/best.pt"

class ObjectDetector:
    def __init__(self, model_path=model_path, conf=0.25):
        try:
            self.model = YOLO(model_path)
            self.model.conf = conf
        except Exception as e:
            print(f"Error when loading model: {e}")
            self.model = None
    
    def object_detects(self, image_path, save_annotated=True):
        if self.model is None:
            return None
        
        results = self.model(image_path)
        detections = []
        result = results[0]
        boxes = result.boxes

        for box in boxes:
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())
            detect_class = self.model.names[class_id]
            xyxy = box.xyxy[0].tolist()
            detections.append({
                "class": detect_class,
                "confidence": confidence,
                "x1": xyxy[0],
                "y1": xyxy[1],
                "x2": xyxy[3],
                "y2": xyxy[3] 
            })

        if save_annotated:
            os.makedirs("static/output", exist_ok=True)
            annotated_image = result.plot()
            output_path = f"static/output/{os.path.basename(image_path)}"
            cv2.imwrite(output_path, annotated_image)

        return detections