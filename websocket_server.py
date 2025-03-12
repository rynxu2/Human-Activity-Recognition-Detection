import websockets
import asyncio
import json
import torch
import numpy as np
from collections import deque
import threading
from train_model import TransformerModel, TransformerModelGrok, TransformerModelChatGPT, TransformerModelDeepseek, TransformerModelChatGPTOptimal, TransformerModelDeepseekOptimal, SEQUENCE_LENGTH, N_FEATURES, HIDDEN_SIZE, NUM_LAYERS, NUM_HEADS
import logging
import warnings
warnings.simplefilter("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealtimeActivityRecognition:
    def __init__(self):
        # Load saved model and preprocessing objects
        checkpoint = torch.load('results\\TransformerModelGrok.pth', map_location=torch.device('cpu'))
        
        # Initialize model
        num_classes = len(checkpoint['label_encoder'].classes_)
        self.model = TransformerModelGrok(
            input_size=N_FEATURES,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            num_heads=NUM_HEADS,
            num_classes=num_classes
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Load preprocessing objects
        self.scaler = checkpoint['scaler']
        self.label_encoder = checkpoint['label_encoder']
        
        # Dictionary to store buffers for each sensor client
        self.sensor_buffers = {}
        
        # Set to store display clients
        self.display_clients = set()
        
        # Initialize lock for thread-safe operations
        self.buffer_lock = threading.Lock()
        
        logger.info("Model loaded successfully")
        logger.info(f"Available activities: {self.label_encoder.classes_}")

    def preprocess_data(self, data):
        try:
            features = np.array([[ 
                float(data['ax']), float(data['ay']), float(data['az']),
                float(data['gx']), float(data['gy']), float(data['gz'])
            ]])
            return self.scaler.transform(features)
        except (KeyError, ValueError) as e:
            logger.error(f"Error preprocessing data: {str(e)}")
            return None

    def make_prediction(self, client_id):
        with self.buffer_lock:
            if client_id not in self.sensor_buffers:
                return None
                
            buffer = self.sensor_buffers[client_id]
            if len(buffer) < SEQUENCE_LENGTH:
                return None
            
            # Convert buffer to numpy array
            sequence = np.array(list(buffer))
            
            # Convert to tensor and add batch dimension
            sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0)
            
            # Make prediction
            with torch.no_grad():
                output = self.model(sequence_tensor)
                probabilities = torch.softmax(output, dim=1)
                predicted_idx = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_idx].item()
                
                predicted_activity = self.label_encoder.inverse_transform([predicted_idx])[0]
                
                return predicted_activity, confidence

    async def send_to_displays(self, prediction_data):
        """Gửi kết quả dự đoán đến tất cả display clients"""
        if not self.display_clients:
            return
        
        message = json.dumps(prediction_data)
        disconnected_clients = set()
        
        for client in self.display_clients:
            try:
                await client.send(message)
            except websockets.exceptions.ConnectionClosed:
                disconnected_clients.add(client)

        # Xóa các client bị mất kết nối
        for client in disconnected_clients:
            self.display_clients.remove(client)
            logger.info(f"Display client disconnected. Remaining: {len(self.display_clients)}")

    async def process_sensor_data(self, websocket, path):
        """Xử lý dữ liệu từ sensor (port 8080)"""
        client_id = id(websocket)
        self.sensor_buffers[client_id] = deque(maxlen=SEQUENCE_LENGTH)
        
        logger.info(f"New sensor client connected. ID: {client_id}")

        try:
            async for message in websocket:
                try:
                    # Parse JSON data
                    data = json.loads(message)
                    
                    # Preprocess data for prediction
                    processed_data = self.preprocess_data(data)
                    if processed_data is None:
                        continue
                    
                    # Add to client's buffer
                    with self.buffer_lock:
                        self.sensor_buffers[client_id].append(processed_data[0])
                    
                    # Make prediction if buffer is full
                    prediction = self.make_prediction(client_id)
                    if prediction:
                        logger.info(f"Sensor {client_id} - Activity: {prediction[0]} (Confidence: {prediction[1]:.2f})")
                        
                        # Gửi dự đoán đến tất cả display clients
                        prediction_data = {
                            "activity": prediction[0],
                            "confidence": float(prediction[1]),
                            "data": data,
                            "sensor_id": client_id
                        }
                        await self.send_to_displays(prediction_data)
                
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON from sensor {client_id}: {str(e)}")
                except Exception as e:
                    logger.error(f"Error processing data from sensor {client_id}: {str(e)}")
        
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Sensor client {client_id} disconnected")
        finally:
            # Cleanup client's buffer
            with self.buffer_lock:
                if client_id in self.sensor_buffers:
                    del self.sensor_buffers[client_id]
            logger.info(f"Sensor client {client_id} removed")

    async def handle_display_client(self, websocket, path):
        """Xử lý display client (port 8081)"""
        client_id = id(websocket)
        self.display_clients.add(websocket)
        
        logger.info(f"New display client connected. ID: {client_id}. Total displays: {len(self.display_clients)}")

        try:
            async for message in websocket:
                pass
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Display client {client_id} disconnected")
        finally:
            self.display_clients.remove(websocket)
            logger.info(f"Display client {client_id} removed. Remaining: {len(self.display_clients)}")

async def main():
    recognition = RealtimeActivityRecognition()
    
    # Server nhận dữ liệu từ sensors (port 8080)
    sensor_server = await websockets.serve(
        recognition.process_sensor_data, "0.0.0.0", 8080
    )
    
    # Server gửi kết quả đến displays (port 8081)
    display_server = await websockets.serve(
        recognition.handle_display_client, "0.0.0.0", 8081
    )

    logger.info("WebSocket servers started:")
    logger.info("- Port 8080: Receiving sensor data")
    logger.info("- Port 8081: Sending predictions to displays")
    
    await asyncio.gather(sensor_server.wait_closed(), display_server.wait_closed())

if __name__ == "__main__":
    asyncio.run(main()) 
