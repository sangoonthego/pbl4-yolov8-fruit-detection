import os
import json
import requests
import time
import atexit
from threading import Thread
from yolov8.detect_object import ObjectDetector
from flask import Flask, jsonify, request
from flask_socketio import SocketIO

snapshot_dir = "snapshots"
BACKEND_UPLOAD_API = "https://fruitstore.loca.lt/api/upload_result"
CONF_THRESHOLD = 0.6  

detector = ObjectDetector()
processed_files = set()

latest_weight = {
    "weight": 0.0
}

weight_ready = {
    "ready": False  
}

def cleanup_snapshots():
    if not os.path.exists(snapshot_dir):
        return
    print("Cleaning up snapshot folder...")
    for file in os.listdir(snapshot_dir):
        try:
            path = os.path.join(snapshot_dir, file)
            if os.path.isfile(path):
                os.remove(path)
        except Exception as e:
            print(f"Failed to delete {file}: {e}")
    print("Snapshot folder cleaned.")

atexit.register(cleanup_snapshots)

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", logger=True, engineio_logger=True)

@socketio.on("connect")
def handle_connect():
    print("Client connected.")

@socketio.on("disconnect")
def handle_disconnect():
    print("Client disconnected.")

@socketio.on("weight_data")
def handle_weight(data):
    try:
        weight = float(data.get("weight", 0))
        latest_weight["weight"] = weight
        weight_ready["ready"] = True
        print(f"⚖ Realtime weight (Socket): {weight:.2f} kg")
        socketio.emit("new_weight", {"weight": weight})
    except Exception as e:
        print(f"Socket weight error: {e}")

@app.route("/weight", methods=["GET", "POST"])
def handle_weight_http():
    if request.method == "POST":
        try:
            data = request.get_json()
            if not data or "weight" not in data:
                return jsonify({"status": "error", "message": "Missing weight"}), 400

            weight = float(data["weight"])
            latest_weight["weight"] = weight
            weight_ready["ready"] = True
            print(f"⚖ Weight via HTTP: {weight:.2f} kg")
            socketio.emit("new_weight", {"weight": weight})

            return jsonify({"status": "success", "weight": weight})
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500

    return jsonify(latest_weight)

def send_to_backend(image_name, result_data):
    image_path = result_data.get("annotated_path")

    if not image_path or not os.path.exists(image_path):
        print(f"No annotated image for {image_name}, skip upload.")
        return

    payload = {
        "image_name": image_name,
        "counts": result_data.get("counts", {}),
        "detections": result_data.get("detections", []),
        "weight": result_data.get("weight", 0)
    }

    print(f"⬆ Uploading result for {image_name}...")

    try:
        with open(image_path, "rb") as f:
            files = {
                "file": f,
                "data": (None, json.dumps(payload, ensure_ascii=False))
            }
            res = requests.post(BACKEND_UPLOAD_API, files=files)

        if res.status_code == 200:
            print("Upload success")
        else:
            print(f"Upload failed: {res.status_code} - {res.text}")

    except Exception as e:
        print(f"Upload error: {e}")

# =============================
# Image processing
# =============================

def process_latest_image():
    if not os.path.exists(snapshot_dir):
        return

    if not weight_ready["ready"]:
        print("⏳ Waiting for first weight...")
        return

    images = [
        f for f in os.listdir(snapshot_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if not images:
        return

    latest_file = max(
        images,
        key=lambda f: os.path.getmtime(os.path.join(snapshot_dir, f))
    )

    if latest_file in processed_files:
        return

    image_path = os.path.join(snapshot_dir, latest_file)
    print(f"\n📸 Processing image: {latest_file}")

    try:
        result = detector.object_detects(image_path)
    except Exception as e:
        print(f"YOLO error: {e}")
        return

    if not isinstance(result, dict):
        print("⚠ YOLO returned invalid result, create empty result")
        result = {
            "counts": {},
            "detections": [],
            "annotated_path": None
        }

    filtered_detections = []
    filtered_counts = {}

    for det in result.get("detections", []):
        conf = det.get("confidence", det.get("conf", 0))

        if conf >= CONF_THRESHOLD:
            filtered_detections.append(det)

            label = det.get("label") or det.get("class_name")
            if label:
                filtered_counts[label] = filtered_counts.get(label, 0) + 1

    if not filtered_detections:
        # print("⚠ No detections with conf >= 0.6 → Skip JSON & upload")
        processed_files.add(latest_file)
        return

    result["detections"] = filtered_detections
    result["counts"] = filtered_counts

    result["weight"] = latest_weight["weight"]
    print(f"⚖ Weight attached: {result['weight']:.2f} kg")

    # =============================
    # Save JSON
    # =============================

    json_path = os.path.splitext(image_path)[0] + ".json"
    try:
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(result, jf, ensure_ascii=False, indent=4)
        print(f"JSON saved: {json_path}")
    except Exception as e:
        print(f"JSON save error: {e}")

    # Upload backend
    send_to_backend(latest_file, result)

    processed_files.add(latest_file)
    print("Done")
    print("-" * 50)

def watcher_loop():
    os.makedirs(snapshot_dir, exist_ok=True)
    processed_files.clear()
    print("👀 Watcher started...")
    while True:
        process_latest_image()
        time.sleep(0.5)

# =============================
# Main
# =============================

if __name__ == "__main__":
    t = Thread(target=watcher_loop, daemon=True)
    t.start()

    print("Flask SocketIO running on port 5000")
    socketio.run(app, host="0.0.0.0", port=5000, debug=True, use_reloader=False)
