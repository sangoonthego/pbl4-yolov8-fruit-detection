import os
import time
import json
import requests
from yolov8.detect_object import ObjectDetector

snapshot_dir = "snapshots"
backend_url = "https://yoursubdomain.loca.lt/api/upload_result"
delay_seconds = 5

detector = ObjectDetector()
processed_files = set()


def send_to_backend(image_name, result):
    image_path = os.path.join(snapshot_dir, image_name)
    annotated_path = result.get("annotated_path")
    detections = result.get("detections", [])
    counts = result.get("counts", {})

    data_json = json.dumps({
        "image_name": image_name,
        "detections": detections,
        "counts": counts
    }, ensure_ascii=False)

    files = {
        "original": open(image_path, "rb"),
        "data": (None, data_json)
    }

    if annotated_path and os.path.exists(annotated_path):
        files["annotated"] = open(annotated_path, "rb")

    try:
        response = requests.post(backend_url, files=files)
        if response.status_code == 200:
            print(f"Sent {image_name} successfully.")
        else:
            print(f"Failed to send {image_name}: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Error sending to backend: {e}")
    finally:
        # Close all opened files
        for f in files.values():
            if hasattr(f, "close"):
                f.close()


def process_new_images():
    files = [f for f in os.listdir(snapshot_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

    for filename in files:
        if filename not in processed_files:
            image_path = os.path.join(snapshot_dir, filename)
            print(f"Detecting objects in {filename} ...")

            result = detector.object_detects(image_path)

            if result is None:
                print(f"Skipped {filename} due to model load error.")
                continue

            json_path = os.path.splitext(image_path)[0] + ".json"
            with open(json_path, "w", encoding="utf-8") as jf:
                json.dump(result, jf, ensure_ascii=False, indent=4)

            send_to_backend(filename, result)
            processed_files.add(filename)
            print(f"Finished processing {filename}\n")


if __name__ == "__main__":
    print("Watching for new images in snapshots directory...")
    os.makedirs(snapshot_dir, exist_ok=True)

    while True:
        process_new_images()
        time.sleep(delay_seconds)
