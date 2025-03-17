#include <WiFi.h>
#include <WebSocketsClient.h>
#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <ArduinoJson.h>

const char* ssid = "TP-LINK_BEF6";
const char* password = "22446688";

// WebSocket server settings
const char* wsHost = "adaptor-cruz-far-road.trycloudflare.com";
const int wsPort = 8080;

WebSocketsClient webSocket;
Adafruit_MPU6050 mpu;

unsigned long lastReadTime = 0;
const int READ_INTERVAL = 20; // 50Hz = reading every 20ms

void webSocketEvent(WStype_t type, uint8_t * payload, size_t length) {
    switch(type) {
        case WStype_DISCONNECTED:
            Serial.println("Disconnected from WebSocket server!");
            break;
        case WStype_CONNECTED:
            Serial.println("Connected to WebSocket server!");
            break;
        case WStype_TEXT:
            // Handle incoming messages if needed
            break;
    }
}

void setup() {
    Serial.begin(115200);
    Wire.begin();
    
    Serial.println("Initializing MPU6050...");
    if (!mpu.begin()) {
        Serial.println("Could not find a valid MPU6050 sensor, check wiring!");
        while (1) {
            delay(10);
        }
    }
    
    Serial.println("MPU6050 Found!");

    WiFi.begin(ssid, password);
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    Serial.println("");
    Serial.println("WiFi connected");
    Serial.println("IP address: ");
    Serial.println(WiFi.localIP());
    
    webSocket.begin(wsHost, wsPort, "/sensor");
    webSocket.onEvent(webSocketEvent);
    webSocket.setReconnectInterval(5000);
}

void loop() {
    webSocket.loop();
    
    unsigned long currentTime = millis();
    if (currentTime - lastReadTime >= READ_INTERVAL) {
        sensors_event_t a, g, temp;
        mpu.getEvent(&a, &g, &temp);
        
        char jsonString[200];
        snprintf(jsonString, sizeof(jsonString),
                "{\"ax\":%.2f,\"ay\":%.2f,\"az\":%.2f,\"gx\":%.2f,\"gy\":%.2f,\"gz\":%.2f}",
                a.acceleration.x, a.acceleration.y, a.acceleration.z,
                g.gyro.x, g.gyro.y, g.gyro.z);
        
        webSocket.sendTXT(jsonString);
        
        lastReadTime = currentTime;
    }
} 