import os
import json
import requests
import time
from yolov8.detect_object import ObjectDetector

snapshot_dir = "snapshots"
backend_url = "https://yoursubdomain.loca.lt/api/upload_result"

detector = ObjectDetector()
processed_files = set()


def send_to_backend(image_name, result_data):
    image_path = result_data.get("annotated_path")
    detections = result_data.get("detections", [])
    counts = result_data.get("counts", {})

    if not image_path or not os.path.exists(image_path):
        print(f"Annotated image not found for {image_name}. Skipping upload.")
        return

    data_json = json.dumps({
        "image_name": image_name,
        "counts": counts,
        "detections": detections
    }, ensure_ascii=False)

    try:
        with open(image_path, "rb") as file:
            files = {
                "file": file,
                "data": (None, data_json)
            }
            response = requests.post(backend_url, files=files)

        if response.status_code == 200:
            print(f"Sent {image_name} successfully.")
        else:
            print(f"Failed to send {image_name}: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Error sending {image_name} to backend: {e}")


def process_latest_image():
    files = [f for f in os.listdir(snapshot_dir)
             if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    if not files:
        return

    latest_file = max(files, key=lambda f: os.path.getmtime(os.path.join(snapshot_dir, f)))

    if latest_file in processed_files:
        return

    image_path = os.path.join(snapshot_dir, latest_file)
    print(f"Detecting objects in: {latest_file}")

    result = detector.object_detects(image_path)
    if result is None:
        print(f"Skipped {latest_file} due to model load error.")
        return

    json_path = os.path.splitext(image_path)[0] + ".json"
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(result, jf, ensure_ascii=False, indent=4)
    print(f"Saved detection result to {json_path}")

    send_to_backend(latest_file, result)
    processed_files.add(latest_file)

    print(f"Finished processing {latest_file}")
    print("Detected object counts:")
    for label, count in result["counts"].items():
        print(f"  {label}: {count}")
    print("-" * 40)


if __name__ == "__main__":
    print("Watching for new snapshots...")
    os.makedirs(snapshot_dir, exist_ok=True)

    while True:
        process_latest_image()
        time.sleep(1)
