import websockets
import asyncio
import json
import torch
import numpy as np
from collections import deque
from datetime import datetime, timezone, timedelta
import threading
from models import TransformerModel, SEQUENCE_LENGTH, N_FEATURES, HIDDEN_SIZE, NUM_LAYERS, NUM_HEADS
import logging
import warnings
from telegram.ext import Application
import socket
warnings.simplefilter("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealtimeActivityRecognition:
    def __init__(self, weightPath):
        self.TeleBotToken = "7698309495:AAEpaxdsF2s76PACe3J7vHB6GPZmoozSf_I"
        self.TeleBotChatID = "-1002319879693"
        self.weightPath = weightPath
        
        checkpoint = torch.load(self.weightPath, map_location=torch.device('cpu'))
        
        num_classes = len(checkpoint['label_encoder'].classes_)
        self.model = TransformerModel(
            input_size=N_FEATURES,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            num_heads=NUM_HEADS,
            num_classes=num_classes 
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        self.scaler = checkpoint['scaler']
        self.label_encoder = checkpoint['label_encoder']
        
        self.sensor_buffers = {}
        self.display_clients = set()
        self.buffer_lock = threading.Lock()
        
        self.previous_activities = {}
        self.consecutive_falls = {}
        self.fall_threshold = 5
        self.last_fall_alert = {}
        self.fall_cooldown = 10
        
        self.bot_app = Application.builder().token(self.TeleBotToken).build()
        self.bot_initialized = False
        self.notification_queue = asyncio.Queue()
        self.notification_task = None
        
        logger.info("Model loaded successfully")
        logger.info(f"Available activities: {self.label_encoder.classes_}")
        
    async def initialize_bot(self):
        if not self.bot_initialized:
            await self.bot_app.initialize()
            self.bot_initialized = True
            
    async def start(self):
        await self.initialize_bot()
        self.notification_task = asyncio.create_task(self.handle_notifications())
        
    async def handle_notifications(self):
        while True:
            try:
                notification = await self.notification_queue.get()
                await self.bot_app.bot.send_message(
                    chat_id=self.TeleBotChatID,
                    text=notification['text'],
                    parse_mode=notification.get('parse_mode', None)
                )
                self.notification_queue.task_done()
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"Error sending notification: {str(e)}")
                
    async def queue_notification(self, text, parse_mode=None):
        await self.notification_queue.put({
            'text': text,
            'parse_mode': parse_mode
        })

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
            
            sequence = np.array(list(buffer))
            
            sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0)
            
            with torch.no_grad():
                output = self.model(sequence_tensor)
                probabilities = torch.softmax(output, dim=1)
                predicted_idx = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_idx].item()
                
                predicted_activity = self.label_encoder.inverse_transform([predicted_idx])[0]
                
                return predicted_activity, confidence

    async def send_to_displays(self, prediction_data):
        if not self.display_clients:
            return
        
        message = json.dumps(prediction_data)
        disconnected_clients = set()
        
        for client in self.display_clients:
            try:
                await client.send(message)
            except websockets.exceptions.ConnectionClosed:
                disconnected_clients.add(client)

        for client in disconnected_clients:
            self.display_clients.remove(client)
            logger.info(f"Display client disconnected. Remaining: {len(self.display_clients)}")
            
    async def send_to_bot(self):
        try:
            utc_plus_7 = timezone(timedelta(hours=7))
            current_time = datetime.now(utc_plus_7).strftime('%Y-%m-%d %H:%M:%S')
            text = (
                "🚨 *CẢNH BÁO NGÃ!* 🚨\n"
                "📌 *Trạng thái:* 🔴 **NGÃ PHÁT HIỆN**\n"
               f"🕒 *Thời gian:* `{current_time}` \n"
                "⚠️ **Hãy kiểm tra ngay!** ⚠️"
            )
            await self.queue_notification(text, parse_mode="Markdown")
        except Exception as e:
            logger.error(f"Error queuing telegram message: {str(e)}")

    async def handle_client(self, websocket, path):
        if path == "/sensor":
            await self.process_sensor_data(websocket)
        elif path == "/display":
            await self.handle_display_client(websocket)
        else:
            logger.warning(f"Unknown client type with path: {path}")
            return

    async def process_sensor_data(self, websocket):
        client_id = id(websocket)
        self.sensor_buffers[client_id] = deque(maxlen=SEQUENCE_LENGTH)
        self.previous_activities[client_id] = None
        self.consecutive_falls[client_id] = 0
        self.last_fall_alert[client_id] = datetime.min.replace(tzinfo=timezone.utc)
        
        logger.info(f"New sensor client connected. ID: {client_id}")

        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    
                    processed_data = self.preprocess_data(data)
                    if processed_data is None:
                        continue
                    
                    with self.buffer_lock:
                        self.sensor_buffers[client_id].append(processed_data[0])
                    
                    prediction = self.make_prediction(client_id)
                    if prediction:
                        activity, confidence = prediction
                        logger.info(f"Sensor {client_id} - Activity: {activity} (Confidence: {confidence:.2f})")
                        
                        current_time = datetime.now(timezone(timedelta(hours=7)))
                        
                        if activity == "falling":
                            self.consecutive_falls[client_id] += 1
                            current_time = datetime.now(timezone(timedelta(hours=7)))
                            if (current_time - self.last_fall_alert[client_id]).total_seconds() > 10:
                                if self.previous_activities[client_id] != "falling":
                                    logger.info(f"Client {client_id} - Gửi cảnh báo đến bot!")
                                    self.previous_activities[client_id] = activity
                                    self.last_fall_alert[client_id] = current_time
                                    await self.send_to_bot()
                        else:
                            self.consecutive_falls[client_id] = 0
                            self.previous_activities[client_id] = activity
                        
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
            with self.buffer_lock:
                if client_id in self.sensor_buffers:
                    del self.sensor_buffers[client_id]
                if client_id in self.previous_activities:
                    del self.previous_activities[client_id]
                if client_id in self.consecutive_falls:
                    del self.consecutive_falls[client_id]
                if client_id in self.last_fall_alert:
                    del self.last_fall_alert[client_id]
            logger.info(f"Sensor client {client_id} removed")

    async def handle_display_client(self, websocket):
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

def get_local_ip():
    """Get the local IP address of the machine"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception as e:
        logger.error(f"Error getting local IP: {e}")
        return "127.0.0.1"

async def main():
    recognition = RealtimeActivityRecognition('results\\StandardScaler\\TransformerModel_w100.pth')
    await recognition.start()
    
    local_ip = get_local_ip()
    port = 8080
    
    server = await websockets.serve(
        recognition.handle_client, local_ip, port
    )
    
    logger.info("WebSocket server started:")
    logger.info(f"- Local Network: ws://{local_ip}:{port}")
    logger.info(f"- Localhost: ws://127.0.0.1:{port}")
    logger.info("- Endpoints: /sensor for sensors, /display for displays")
    
    try:
        await server.wait_closed()
    finally:
        if recognition.notification_task:
            recognition.notification_task.cancel()

if __name__ == "__main__":
    asyncio.run(main())