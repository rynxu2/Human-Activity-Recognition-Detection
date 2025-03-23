# 🎓 Human Activity Recognition - MPU6050

<div align="center">

<p align="center">
  <img src="https://github.com/drkhanusa/DNU_PlagiarismChecker/raw/main/docs/images/logo.png" alt="DaiNam University Logo" width="200"/>
</p>

</div>

<h3 align="center">🔬 Human Activity Recognition Through AI</h3>

<p align="center">
  <strong>A Real-Time Activity Detection System Using MPU6050 Sensor and Deep Learning</strong>
</p>

<p align="center">
  <a href="#-architecture">Architecture</a> •
  <a href="#-key-features">Features</a> •
  <a href="#-tech-stack">Tech Stack</a> •
  <a href="#-installation">Installation</a> •
  <a href="#-getting-started">Getting Started</a> •
  <a href="#-documentation">Docs</a>
</p>

## 🏗️ Architecture

<p align="center">
  <img src="https://i.postimg.cc/CxZvMTNF/Blank-diagram.png" alt="System Architecture" width="900"/>
</p>

The system employs a three-tier architecture:

1. **📱 Data Collection Layer**: Captures real-time motion data using MPU6050 sensor
2. **🔄 Processing Layer**: Processes sensor data and classifies activities using AI
3. **📊 Visualization Layer**: Real-time dashboard for activity monitoring

## ✨ Key Features

### 🧠 AI-Powered Activity Recognition
- **Real-time Processing**: Instant classification of human activities
- **Transformer-based Model**: High accuracy using advanced deep learning
- **Multi-activity Support**: Walking, Jogging, Sitting, Standing, Falling detection
- **Abnormal Activity Alerts**: Automatic fall detection and notifications

### ⚡ High-Performance System
- **WebSocket Protocol**: Low-latency real-time data streaming
- **Optimized Data Handling**: Efficient sensor data buffering and processing
- **Scalable & Flexible**: Supports multiple sensors and future expansions

### 📊 Smart Monitoring & Alerts
- **Live Sensor Charts**: Real-time acceleration and gyroscope data display
- **Activity Dashboard**: Current activity status and confidence levels
- **Telegram Notifications**: Instant alerts for abnormal activities like falls

## 🔧 Tech Stack

<div align="center">

### Core Technologies
[![React](https://img.shields.io/badge/React-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://reactjs.org/)
[![WebSocket](https://img.shields.io/badge/WebSocket-010101?style=for-the-badge&logo=socket.io&logoColor=white)](https://developer.mozilla.org/en-US/docs/Web/API/WebSockets_API)
[![PtTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://www.tensorflow.org/)

### Hardware Components
[![ESP32](https://img.shields.io/badge/ESP32-E7352C?style=for-the-badge&logo=espressif&logoColor=white)](https://www.espressif.com/)
[![MPU6050](https://img.shields.io/badge/MPU6050-00979D?style=for-the-badge&logo=arduino&logoColor=white)](https://invensense.tdk.com/products/motion-tracking/6-axis/mpu-6050/)

</div>

## 📥 Installation

### 🛠️ Prerequisites

- 🔌 **ESP32** - Microcontroller
- 🎯 **MPU6050** - 6-axis motion sensor
- 📱 **Node.js** `14+` - For web dashboard
- 🐍 **Python** `3.9+`- Core programming language

### ⚙️ Project Setup

1. 📦 **Hardware Assembly**
   ```
   Connect MPU6050 to ESP32:
   - VCC → 3.3V
   - GND → GND
   - SCL → GPIO22
   - SDA → GPIO21
   ```

2. 🌟 **ESP32 Setup**
   ```bash
   # Install required libraries in Arduino IDE
   - ESP32 Board
   - WebSockets
   - ArduinoJson
   - Adafruit MPU6050
   ```

3. 🐍 **Websocket Setup**
   ```bash
   # Install required libraries in Arduino IDE
   pip install -r requirements.txt
   ```

4. 📱 **Website Setup**
   ```bash
   # Navigate to website directory
   cd website

   # Install dependencies
   npm install

   # Change websocket url
   notepad src/App.js
   change websocket url
   ```

## 🚀 Getting Started

### ⚡ Quick Start
1. Clone repository
   ```bash
   git clone https://github.com/rynxu2/Human-Activity-Recognition-Detection.git
   cd Human-Activity-Recognition-Detection
   ```
2. Upload ESP32 code
3. Cài đặt các thư viện Python cần thiết
   ```bash
   pip install -r requirements.txt
   ```
4. Start websocket server
   ```bash
   python websocket_server.py
   ```
5. Start the web dashboard
   ```bash
   cd website
   npm run start
   ```
6. Access dashboard at `http://localhost:3000`

### 🖥️ Train Model

```bash
models.ipynb
```
---