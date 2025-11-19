from flask import Flask, jsonify, request
from flask_socketio import SocketIO

# =============================
# Cấu hình Ứng dụng Flask
# =============================
app = Flask(__name__)
# Đặt logger=True, engineio_logger=True để in log chi tiết
# Sử dụng cors_allowed_origins="*" cho phép mọi client kết nối SocketIO
socketio = SocketIO(app, cors_allowed_origins="*", logger=True, engineio_logger=True)

# Lưu dữ liệu cân mới nhất (sử dụng global để dễ truy cập)
latest_weight = {"weight": 0.0}

# =============================
# SocketIO Events (Dành cho giao tiếp Realtime)
# =============================
@socketio.on('connect')
def handle_connect():
    """Xử lý khi một client SocketIO kết nối."""
    print("🔗 [SocketIO] Client connected!")

@socketio.on('disconnect')
def handle_disconnect():
    """Xử lý khi một client SocketIO ngắt kết nối."""
    print("❌ [SocketIO] Client disconnected!")

@socketio.on('weight_data')
def handle_weight(data):
    """
    Xử lý dữ liệu cân được gửi từ client (ví dụ: ESP8266) qua SocketIO.
    Client gửi JSON dạng: {"weight": 1.25}
    """
    try:
        # Lấy giá trị cân nặng và chuyển đổi sang float
        weight = float(data.get("weight", 0))
        
        # Cập nhật giá trị cân mới nhất
        latest_weight["weight"] = weight
        
        print(f"⚖️ [Realtime] Dữ liệu cân nhận (SocketIO): {weight:.2f} kg")
        
        # Phát dữ liệu tới các client web khác đang lắng nghe
        socketio.emit('new_weight', {"weight": weight})
        
    except Exception as e:
        print(f"❌ [Error] Lỗi khi xử lý dữ liệu cân qua SocketIO: {e}")

# =============================
# HTTP Endpoint /weight (Dành cho giao tiếp REST)
# =============================
@app.route("/weight", methods=["GET", "POST"]) # <--- SỬA ĐỔI: Chấp nhận cả GET và POST
def handle_weight_http():
    """
    Xử lý yêu cầu HTTP GET (lấy dữ liệu) và POST (cập nhật dữ liệu).
    """
    
    if request.method == "POST":
        # === Xử lý POST (Nhận dữ liệu cân) ===
        try:
            # Đọc dữ liệu JSON từ yêu cầu
            data = request.get_json()
            if not data or 'weight' not in data:
                return jsonify({"status": "error", "message": "Missing 'weight' in JSON payload"}), 400

            weight = float(data['weight'])
            
            # Cập nhật giá trị cân mới nhất
            global latest_weight
            latest_weight["weight"] = weight
            
            print(f"📡 [HTTP POST /weight] Nhận cân: {weight:.2f} kg")
            
            # Phát dữ liệu tới các client SocketIO khác (nếu muốn)
            socketio.emit('new_weight', {"weight": weight})
            
            return jsonify({"status": "success", "weight": weight}), 200
            
        except ValueError:
            return jsonify({"status": "error", "message": "Invalid weight format"}), 400
        except Exception as e:
            print(f"❌ [Error] Lỗi khi xử lý POST /weight: {e}")
            return jsonify({"status": "error", "message": str(e)}), 500

    elif request.method == "GET":
        # === Xử lý GET (Trả về dữ liệu cân mới nhất) ===
        weight = latest_weight.get("weight", 0.0)
        print(f"📡 [HTTP GET /weight] Trả về cân: {weight:.2f} kg")
        
        # Trả về dữ liệu cân nặng dưới dạng JSON
        return jsonify(latest_weight)

# =============================
# Chạy Flask SocketIO server
# =============================
if __name__ == '__main__':
    print("==============================================")
    print("🚀 Flask SocketIO server is running on port 5000")
    print("==============================================")
    # debug=True để in log chi tiết (nhưng log của SocketIO đã bật)
    # use_reloader=False để tránh server chạy 2 lần
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, use_reloader=False)