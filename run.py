import websockets
from app import app, socketio
from websocket_server import RealtimeActivityRecognition
import asyncio
import threading
import logging
import socket

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def find_free_port(start_port, max_attempts=10):
    """Find an available port starting from start_port"""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('127.0.0.1', port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"Could not find a free port after {max_attempts} attempts")

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

def run_websocket():
    async def start_websocket():
        recognition = RealtimeActivityRecognition('results/StandardScaler/TransformerModel_w100.pth')
        await recognition.start()
        
        # Find available port
        local_ip = get_local_ip()
        
        websocket_port = find_free_port(8080)
        
        server = await websockets.serve(
            recognition.handle_client, local_ip, websocket_port
        )
        
        logger.info("WebSocket server started:")
        logger.info(f"- Local Network: ws://{local_ip}:{websocket_port}")
        logger.info(f"- Localhost: ws://127.0.0.1:{websocket_port}")
        logger.info("- Endpoints: /sensor for sensors, /display for displays")
        
        try:
            await server.wait_closed()
        finally:
            if recognition.notification_task:
                recognition.notification_task.cancel()

    asyncio.run(start_websocket())

if __name__ == '__main__':
    try:
        # Get local IP
        local_ip = get_local_ip()
        
        # First check if ports are available
        flask_port = find_free_port(5000)
        
        # Start WebSocket server
        websocket_thread = threading.Thread(target=run_websocket)
        websocket_thread.daemon = True
        websocket_thread.start()
        
        # Start Flask application
        logger.info("Flask server started:")
        logger.info(f"- Local Network: http://{local_ip}:{flask_port}")
        logger.info(f"- Localhost: http://127.0.0.1:{flask_port}")
        
        socketio.run(
            app, 
            host=local_ip,
            port=flask_port, 
            debug=False, 
            allow_unsafe_werkzeug=True, 
            use_reloader=False
        )
    except Exception as e:
        logger.error(f"Failed to start servers: {e}")
        raise