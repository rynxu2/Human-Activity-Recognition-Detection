import websocket

# URL của WebSocket Server
WS_URL = "wss://websocket-server-production-ebad.up.railway.app"  # Thay URL server của bạn

# Xử lý khi kết nối thành công
def on_open(ws):
    print("✅ Kết nối WebSocket thành công!")
    ws.send("Hello Server!")

# Xử lý khi nhận dữ liệu từ server
def on_message(ws, message):
    print(f"📩 Nhận từ server: {message}")

# Xử lý khi gặp lỗi
def on_error(ws, error):
    print(f"❌ Lỗi WebSocket: {error}")

# Xử lý khi đóng kết nối
def on_close(ws, close_status_code, close_msg):
    print("🔌 WebSocket đóng kết nối")

# Tạo WebSocket Client
ws = websocket.WebSocketApp(
    WS_URL,
    on_open=on_open,
    on_message=on_message,
    on_error=on_error,
    on_close=on_close
)

# Chạy WebSocket Client
ws.run_forever()
