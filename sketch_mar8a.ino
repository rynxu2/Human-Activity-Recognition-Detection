#include <WiFi.h>
#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <WebSocketsClient.h>

// --- Cấu hình WiFi ---
const char* ssid = "WIFI GIANG VIEN";
const char* password = "dhdn7799";

// --- Cấu hình WebSocket ---
const char* ws_server = "unions-singles-custom-participants.trycloudflare.com";
const int ws_port = 8080;
WebSocketsClient webSocket;

// --- Cấu trúc dữ liệu cảm biến ---
struct SensorData {
  float ax, ay, az;
  float gx, gy, gz;
};

// --- Cấu hình buffer vòng ---
#define BUFFER_SIZE 100  // Sử dụng 100 để tiết kiệm bộ nhớ
SensorData dataBuffer[BUFFER_SIZE];
int bufferHead = 0;
int bufferTail = 0;
int bufferedItems = 0;

// --- Biến đếm số bản ghi đã gửi ---
unsigned long sentRecords = 0;

// --- Quản lý thời gian ---
unsigned long lastSampleTime = 0;
const unsigned int SAMPLE_INTERVAL = 20;  // 50Hz (20ms)
unsigned long lastReconnectAttempt = 0;
const unsigned int RECONNECT_INTERVAL = 5000;

// --- Quản lý ACK ---
bool awaitingAck = false;
unsigned long lastSendTime = 0;
const unsigned int ACK_TIMEOUT = 1000;

Adafruit_MPU6050 mpu;

// --- Hàm thiết lập WiFi ---
void setupWiFi() {
  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nConnected! IP: " + WiFi.localIP().toString());
}

// --- Hàm thiết lập MPU6050 ---
void setupMPU() {
  if (!mpu.begin(0x68)) {
    Serial.println("Failed to initialize MPU6050!");
    while (1) delay(10);
  }
  mpu.setGyroRange(MPU6050_RANGE_250_DEG);
  mpu.setAccelerometerRange(MPU6050_RANGE_4_G);
  mpu.setFilterBandwidth(MPU6050_BAND_44_HZ);
  Serial.println("MPU6050 initialized!");
}

// --- Kết nối WebSocket ---
void connectWebSocket() {
  webSocket.begin(ws_server, ws_port, "/");
  webSocket.onEvent(webSocketEvent);
  webSocket.setReconnectInterval(RECONNECT_INTERVAL);
}

// --- Xử lý sự kiện WebSocket ---
void webSocketEvent(WStype_t type, uint8_t *payload, size_t length) {
  switch (type) {
    case WStype_DISCONNECTED:
      Serial.println("WebSocket disconnected");
      break;
    case WStype_CONNECTED:
      Serial.println("WebSocket connected");
      sendBufferedData();
      break;
    case WStype_TEXT:
      if (strncmp((char*)payload, "ACK", 3) == 0) {
        awaitingAck = false;
        Serial.println("ACK received");
      }
      break;
    default:
      break;
  }
}

// --- Lưu dữ liệu vào buffer ---
void bufferSensorData(const sensors_event_t &a, const sensors_event_t &g) {
  if (bufferedItems >= BUFFER_SIZE) {  // Nếu buffer đầy, ghi đè dữ liệu cũ
    bufferTail = (bufferTail + 1) % BUFFER_SIZE;
    bufferedItems--;
  }
  dataBuffer[bufferHead] = { a.acceleration.x, a.acceleration.y, a.acceleration.z,
                             g.gyro.x, g.gyro.y, g.gyro.z };
  bufferHead = (bufferHead + 1) % BUFFER_SIZE;
  bufferedItems++;
}

// --- Đóng gói và gửi dữ liệu nhị phân ---
void sendBinaryData(const SensorData &data) {
  if (!webSocket.isConnected()) return;
  uint8_t packet[24];
  memcpy(packet,     &data.ax, 4);
  memcpy(packet + 4, &data.ay, 4);
  memcpy(packet + 8, &data.az, 4);
  memcpy(packet + 12, &data.gx, 4);
  memcpy(packet + 16, &data.gy, 4);
  memcpy(packet + 20, &data.gz, 4);
  webSocket.sendBIN(packet, sizeof(packet));
}

// --- Gửi dữ liệu từ buffer ---
void sendSensorData() {
  static unsigned long lastSend = 0;
  if (!webSocket.isConnected() || millis() - lastSend < SAMPLE_INTERVAL || bufferedItems <= 0)
    return;
  
  SensorData currentData = dataBuffer[bufferTail];
  sendBinaryData(currentData);
  bufferTail = (bufferTail + 1) % BUFFER_SIZE;
  bufferedItems--;
  awaitingAck = true;
  lastSend = millis();
  lastSendTime = millis();
  sentRecords++;  // Tăng số bản ghi đã gửi
  Serial.printf("Sent record: %lu, remaining buffer: %d\n", sentRecords, bufferedItems);
}

// --- Gửi lại dữ liệu trong buffer ---
void sendBufferedData() {
  int sendCount = 0;
  while (bufferedItems > 0 && webSocket.isConnected() && !awaitingAck && sendCount < 50) {
    sendSensorData();
    sendCount++;
    yield();
  }
}

// --- Thu thập và xử lý dữ liệu cảm biến ---
void handleSensorData() {
  if (millis() - lastSampleTime >= SAMPLE_INTERVAL) {
    lastSampleTime = millis();
    sensors_event_t a, g, temp;
    if (mpu.getEvent(&a, &g, &temp)) {
      bufferSensorData(a, g);
      sendSensorData();
    } else {
      Serial.println("Failed to get sensor data");
    }
  }
}

// --- Xử lý timeout ACK ---
void handleAckTimeout() {
  if (awaitingAck && (millis() - lastSendTime > ACK_TIMEOUT)) {
    Serial.println("ACK timeout, resetting flag");
    awaitingAck = false;
  }
}

// --- Xử lý kết nối lại ---
void handleReconnection() {
  if (!webSocket.isConnected() && (millis() - lastReconnectAttempt > RECONNECT_INTERVAL)) {
    Serial.println("Attempting WebSocket reconnect...");
    connectWebSocket();
    lastReconnectAttempt = millis();
  }
}

void setup() {
  Serial.begin(115200);
  setupWiFi();
  WiFi.setAutoReconnect(true);
  WiFi.persistent(true);
  
  setupMPU();
  connectWebSocket();
}

void loop() {
  webSocket.loop();
  
  // Kiểm tra và log trạng thái hệ thống mỗi 5 giây
  static unsigned long lastCheck = 0;
  if (millis() - lastCheck > 5000) {
    lastCheck = millis();
    if (!webSocket.isConnected()) {
      Serial.println("Attempting reconnect...");
      connectWebSocket();
    }
    Serial.printf("Status: Buffer %d/%d, WiFi RSSI: %ddBm, Heap: %d\n",
                  bufferedItems, BUFFER_SIZE, WiFi.RSSI(), ESP.getFreeHeap());
  }
  
  handleSensorData();
  handleReconnection();
  handleAckTimeout();
}
