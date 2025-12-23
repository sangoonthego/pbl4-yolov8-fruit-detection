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

detector = ObjectDetector()
processed_files = set()
latest_weight = {"weight": 0.0}

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

@socketio.on('connect')
def handle_connect():
    print("Client connected.")

@socketio.on('disconnect')
def handle_disconnect():
    print("Client disconnected.")

@socketio.on('weight_data')
def handle_weight(data):
    try:
        weight = float(data.get("weight", 0))
        latest_weight["weight"] = weight
        print(f"Realtime weight: {weight:.2f} kg")
        socketio.emit('new_weight', {"weight": weight})
    except Exception as e:
        print(f"Error processing weight via SocketIO: {e}")

@app.route("/weight", methods=["GET", "POST"])
def handle_weight_http():
    if request.method == "POST":
        try:
            data = request.get_json()
            if not data or 'weight' not in data:
                return jsonify({"status": "error", "message": "Missing 'weight'"}), 400
            weight = float(data['weight'])
            latest_weight["weight"] = weight
            print(f"Weight received via HTTP: {weight:.2f} kg")
            socketio.emit('new_weight', {"weight": weight})
            return jsonify({"status": "success", "weight": weight}), 200
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500
    else:
        return jsonify(latest_weight)

def send_to_backend(image_name, result_data):
    image_path = result_data.get("annotated_path")
    if not image_path or not os.path.exists(image_path):
        print(f"Annotated image not found for {image_name}, skipping upload.")
        return
    
    payload_json = json.dumps({
        "image_name": image_name,
        "counts": result_data.get("counts", {}),
        "detections": result_data.get("detections", []),
        "weight": result_data.get("weight", 0)
    }, ensure_ascii=False)

    print(f"Uploading result for {image_name}...")

    try:
        with open(image_path, "rb") as f:
            files = {
                "file": f,
                "data": (None, payload_json)
            }
            res = requests.post(BACKEND_UPLOAD_API, files=files)
            if res.status_code == 200:
                print(f"Upload completed: {image_name}")
            else:
                print(f"Upload failed: {res.status_code} - {res.text}")
    except Exception as e:
        print(f"Error sending to backend: {e}")

def process_latest_image():
    files = [
        f for f in os.listdir(snapshot_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    if not files:
        return

    latest_file = max(files, key=lambda f: os.path.getmtime(os.path.join(snapshot_dir, f)))
    if latest_file in processed_files:
        return

    # Wait until we have valid weight
    if latest_weight["weight"] == 0.0:
        print("Waiting for weight data...")
        return

    image_path = os.path.join(snapshot_dir, latest_file)
    print(f"Processing latest snapshot: {latest_file}")
    
    result = detector.object_detects(image_path)
    if result is None:
        print("Model returned None, skipping.")
        return

    result["weight"] = latest_weight["weight"]
    print(f"Attached realtime weight: {result['weight']:.2f} kg")

    json_path = os.path.splitext(image_path)[0] + ".json"
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(result, jf, ensure_ascii=False, indent=4)
    print(f"Saved JSON: {json_path}")

    send_to_backend(latest_file, result)
    processed_files.add(latest_file)
    print(f"Completed processing: {latest_file}")
    print("-------------------------------------------")

def watcher_loop():
    os.makedirs(snapshot_dir, exist_ok=True)
    print("Watcher started, monitoring snapshots folder...")
    while True:
        process_latest_image()
        time.sleep(0.5)  


if __name__ == "__main__":
    t = Thread(target=watcher_loop, daemon=True)
    t.start()
    print("Flask SocketIO server running on port 5000")
    socketio.run(app, host="0.0.0.0", port=5000, debug=True, use_reloader=False)