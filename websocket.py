import asyncio
import websockets
import struct
import signal
import sys
import torch
import numpy as np
from collections import deque
import torch.nn as nn

HOST = "0.0.0.0"
PORT = 8080
WINDOW_SIZE = 50
STEP_SIZE = 10
received_count = 0
data_buffer = deque(maxlen=WINDOW_SIZE)  

received_count = 0

class HARTransformer(nn.Module):
    def __init__(self, input_dim, num_classes, num_heads=4, num_layers=2, hidden_dim=128):
        super(HARTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=0.1),
            num_layers=num_layers
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x) 
        x = x.permute(1, 0, 2) 
        x = self.transformer(x)  
        x = x.permute(1, 0, 2)
        x = x.mean(dim=1)
        x = self.fc(x)
        return x

def load_transformer_model(model_path):
    try:
        model = torch.load(model_path, map_location=torch.device('cpu'))
        model.eval()
        print("✅ Loaded Transformer model successfully")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

# Tiền xử lý dữ liệu
def preprocess_data(data_window):
    if len(data_window) < WINDOW_SIZE:
        return None
    
    # Chuyển dữ liệu thành numpy array
    data_array = np.array(data_window)
    
    # Chuẩn hóa dữ liệu (có thể điều chỉnh theo preprocessing của bạn)
    mean = np.mean(data_array, axis=0)
    std = np.std(data_array, axis=0)
    normalized_data = (data_array - mean) / (std + 1e-8)
    
    # Chuyển thành tensor
    data_tensor = torch.FloatTensor(normalized_data).unsqueeze(0)  # Thêm batch dimension
    return data_tensor

# Hàm dự đoán hoạt động
def predict_activity(model, data_tensor):
    with torch.no_grad():
        outputs = model(data_tensor)
        # Giả sử output là logits cho các lớp hoạt động
        predicted = torch.argmax(outputs, dim=-1)
        return predicted.item()
    
ACTIVITY_LABELS = {
    0: "walking",
    1: "jogging",
    2: "sitting",
    3: "standing",
    4: "jumping",
    5: "falling"
}

async def handler(websocket, path):
    global received_count
    print("Client connected")

    model = HARTransformer(input_dim=6, num_classes=6)
    state_dict = torch.load("best_model.pth", map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    
    try:
        async for data in websocket:
            if len(data) == 24:
                ax, ay, az, gx, gy, gz = struct.unpack('ffffff', data)
                received_count += 1
                
                data_buffer.append([ax, ay, az, gx, gy, gz])
                
                if len(data_buffer) == WINDOW_SIZE and received_count % STEP_SIZE == 0:
                    # Tiền xử lý dữ liệu
                    input_tensor = preprocess_data(data_buffer)
                    if input_tensor is not None:
                        # Dự đoán hoạt động
                        activity_idx = predict_activity(model, input_tensor)
                        activity_name = ACTIVITY_LABELS.get(activity_idx, "unknown")
                        
                        # In kết quả dự đoán
                        print(f"🤖 Predicted Activity: {activity_name}")
                        
                        # Gửi kết quả về client
                        response = {
                            "status": "success",
                            "activity": activity_name,
                            "confidence": 1.0  # Có thể thêm confidence score nếu cần
                        }
                        await websocket.send(str(response))
                else:
                    await websocket.send("ACK")
    except websockets.ConnectionClosed:
        print("Client disconnected")
    except Exception as e:
        print(f"⚠️ Error: {e}")

async def main():
    try:
        server = await websockets.serve(handler, HOST, PORT)
        print(f"✅ WebSocket server running on ws://{HOST}:{PORT}")
        await server.wait_closed()
    except Exception as e:
        print(f"❌ Server error: {e}")

# 📌 Xử lý khi nhấn Ctrl + C
def signal_handler(sig, frame):
    print("\n🛑 Stopping server...")
    loop = asyncio.get_event_loop()
    loop.stop()
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    loop = asyncio.get_event_loop()
    loop.set_exception_handler(lambda loop, context: print(f"⚠️ Exception: {context['message']}"))
    
    try:
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        print("🛑 Server stopped by user.")
    except Exception as e:
        print(f"❌ Critical error: {e}")
    finally:
        loop.close()
