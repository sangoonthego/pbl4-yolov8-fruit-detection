import cv2
import urllib.request
import numpy as np
import time
import os

# URL snapshot of ESP32-CAM
esp_url = "http://192.168.1.21/cam-lo.jpg"

save_folder = "snapshots"
os.makedirs(save_folder, exist_ok=True) 

count = 0
while True:
    try:
        img_resp = urllib.request.urlopen(esp_url)
        img_np = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        
        if frame is not None:
            filename = os.path.join(save_folder, f"snapshot_{count:03d}.jpg")
            cv2.imwrite(filename, frame)
            print(f"Saved: {filename}")

            cv2.imshow("ESP32 Snapshot", frame)
            count += 1
        else:
            print("Cannot Read frame")

    except Exception as e:
        print("Error:", e)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    time.sleep(5)

cv2.destroyAllWindows()