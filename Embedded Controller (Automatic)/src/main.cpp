// ---------------- Pin control (5 PWM-like + 1 digital) ---------------- //
// Pin Channels:
//  1) squeeze_plate  (PWM 0..1023)
//  2) ion_source     (PWM 0..1023)
//  3) wein_filter    (PWM 0..1023)
//  4) cone_1         (PWM 0..1023)
//  5) cone_2         (PWM 0..1023)
//
//  6) switch_logic   (digital 0/1)

#include <Arduino.h>
#include "log.h"
#include <Wire.h>
#include <WiFi.h>
#include <ESPmDNS.h>
#include <WiFiUdp.h>
#include <ArduinoOTA.h>
#include <Adafruit_ADS1X15.h>
#include <Adafruit_MCP4725.h>

Adafruit_ADS1115 ads;
Adafruit_MCP4725 dac;

const int LED_PIN = 2; 

float TargetVoltage = 0.0;
unsigned long lastHeartbeat = 0;

#ifndef SQUEEZE_PLATE_PIN
#define SQUEEZE_PLATE_PIN 25  
#endif
#ifndef ION_SOURCE_PIN
#define ION_SOURCE_PIN 26     
#endif
#ifndef WEIN_FILTER_PIN
#define WEIN_FILTER_PIN 27   
#endif
#ifndef CONE_1_PIN
#define CONE_1_PIN 32        
#endif
#ifndef CONE_2_PIN
#define CONE_2_PIN 33       
#endif
#ifndef SWITCH_LOGIC_PIN
#define SWITCH_LOGIC_PIN 16   
#endif

const int CHANNEL_PINS[6] = {SQUEEZE_PLATE_PIN, ION_SOURCE_PIN, WEIN_FILTER_PIN, CONE_1_PIN, CONE_2_PIN, SWITCH_LOGIC_PIN};
int channelValues[6] = {0, 0, 0, 0, 0, 0};

static const int PWM_FREQ = 5000;         
static const int PWM_RESOLUTION = 10;     
static const int PWM_MAX = (1 << PWM_RESOLUTION) - 1;
static const int LEDC_CHANNELS[5] = {0, 1, 2, 3, 4};

void initChannels() {
  
  for (int i = 0; i < 5; i++) {
    pinMode(CHANNEL_PINS[i], OUTPUT);
    ledcSetup(LEDC_CHANNELS[i], PWM_FREQ, PWM_RESOLUTION);
    ledcAttachPin(CHANNEL_PINS[i], LEDC_CHANNELS[i]);
    ledcWrite(LEDC_CHANNELS[i], 0);
    channelValues[i] = 0;
  }
  
  pinMode(CHANNEL_PINS[5], OUTPUT);
  digitalWrite(CHANNEL_PINS[5], LOW);
  channelValues[5] = 0;
}

void setChannel(uint8_t idx1, int value) {
  if (idx1 < 1 || idx1 > 6) {
    Serial.println("ERROR: PIN index out of range!");
    return;
  }
  uint8_t i = idx1 - 1;
  int pin = CHANNEL_PINS[i];
  if (i < 5) {
    if (value < 0) value = 0;
    if (value > PWM_MAX) value = PWM_MAX;
    ledcWrite(LEDC_CHANNELS[i], value);
    channelValues[i] = value;
    Serial.println(String("ACK PIN ") + idx1 + " " + value);
  } else {
    int v = value ? HIGH : LOW;
    digitalWrite(pin, v);
    channelValues[i] = value ? 1 : 0;
    Serial.println(String("ACK PIN ") + idx1 + " " + channelValues[i]);
  }
}

void reportChannels() {
  Serial.print("PINS ");
  Serial.print("squeeze_plate="); Serial.print(channelValues[0]); Serial.print(" ");
  Serial.print("ion_source=");    Serial.print(channelValues[1]); Serial.print(" ");
  Serial.print("wein_filter=");   Serial.print(channelValues[2]); Serial.print(" ");
  Serial.print("cone_1=");        Serial.print(channelValues[3]); Serial.print(" ");
  Serial.print("cone_2=");        Serial.print(channelValues[4]); Serial.print(" ");
  Serial.print("switch_logic=");  Serial.print(channelValues[5]);
  Serial.println("");
}

void blinkOnce(int duration = 100) {
  digitalWrite(LED_PIN, HIGH);
  delay(duration);
  digitalWrite(LED_PIN, LOW);
}

void blinkError(int flashes = 3, int duration = 100) {
  for (int i = 0; i < flashes; i++) {
    digitalWrite(LED_PIN, HIGH);
    delay(duration);
    digitalWrite(LED_PIN, LOW);
    delay(duration);
  }
}

void setup() {
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, HIGH);

  Serial.begin(115200);
  delay(1500);

  LOG_INFO("System initialising...");

  WiFi.begin("YourSSID", "YourPassword");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi connected, IP address:");
  Serial.println(WiFi.localIP());
  
  ArduinoOTA.setHostname("esp32dev");

  ArduinoOTA.onStart([]() {
    Serial.println("OTA connection starting...");
  });

  ArduinoOTA.onEnd([]() {
    Serial.println("OTA connection established...");
  });

  ArduinoOTA.onProgress([](unsigned int progress, unsigned int total) {
    if (total) {
      Serial.printf("OTA Progress: %u%%\r", (progress * 100) / total);
    }
  });

  ArduinoOTA.onError([](ota_error_t error) {
    Serial.printf("Error[%u]: ", error);
    if (error == OTA_AUTH_ERROR) Serial.println("Authentication Failed...");
    else if (error == OTA_BEGIN_ERROR) Serial.println("Begin Failed...");
    else if (error == OTA_CONNECT_ERROR) Serial.println("Connect Failed...");
    else if (error == OTA_RECEIVE_ERROR) Serial.println("Receive Failed...");
    else if (error == OTA_END_ERROR) Serial.println("End Failed...");
  });

  ArduinoOTA.begin();
  Serial.println("Ready for OTA updates...");

  Wire.begin();

  if (!ads.begin()) {
    LOG_ERROR("ADS1115 not found!");
    blinkError(5);
  } else {
    LOG_INFO("ADS1115 ready...");
  }

  if (!dac.begin(0x60)) {
    LOG_ERROR("MCP4725 not found!");
    blinkError(5);
  } else {
    LOG_INFO("MCP4725 ready...");
  }

  LOG_INFO("Setup complete. Awaiting commands...");
  digitalWrite(LED_PIN, LOW);

  initChannels();
}

void SetVoltage(float volts) {
  if (volts < 0) volts = 0;
  if (volts > 3.3) volts = 3.3;

  uint16_t dacValue = (uint16_t)((volts / 3.3) * 4095.0);
  dac.setVoltage(dacValue, false);
  TargetVoltage = volts;

  Serial.println(String("DAC set to ") + volts + " V");
  blinkOnce();  
}

float ReadVoltage() {
  int16_t raw_voltage = ads.readADC_SingleEnded(0);
  return ads.computeVolts(raw_voltage);
}

void loop() {
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();

    if (cmd.startsWith("SET")) {
      float voltage_target = cmd.substring(4).toFloat();
      SetVoltage(voltage_target);
      Serial.println("ACK SET");
    }
    else if (cmd.startsWith("PIN")) {
      int s1 = cmd.indexOf(' ');
      int s2 = cmd.indexOf(' ', s1 + 1);
      if (s1 > 0 && s2 > s1) {
        String token = cmd.substring(s1 + 1, s2);
        token.trim();
        uint8_t idx = 0;
        String t = token; t.toLowerCase();

        if (t == "squeeze_plate") idx = 1;
        else if (t == "ion_source") idx = 2;
        else if (t == "wein_filter") idx = 3;
        else if (t == "cone_1") idx = 4;
        else if (t == "cone_2") idx = 5;
        else if (t == "switch_logic") idx = 6;
        else idx = (uint8_t)token.toInt();

        int val = cmd.substring(s2 + 1).toInt();
        setChannel(idx, val);
      } else {
        Serial.println("ERROR: PIN syntax");
      }
    }
    else if (cmd.equalsIgnoreCase("GET PINS") || cmd.equalsIgnoreCase("PINS")) {
      reportChannels();
    }
    else if (cmd.equalsIgnoreCase("READ")) {
      float live_voltage = ReadVoltage();
      Serial.println(String("MEASURED ") + live_voltage + " V");
      blinkOnce(80);
    }
    else if (cmd.equalsIgnoreCase("PING")) {
      Serial.println("OK");
      blinkOnce(50);
    }
    else {
      Serial.println("ERROR: Unknown command");
      blinkError(2, 70);
    }
  }

  float v = ReadVoltage();
  Serial.println(String("MEASURED ") + v + " V");

  if (millis() - lastHeartbeat > 2000) {
    digitalWrite(LED_PIN, HIGH);
    delay(50);
    digitalWrite(LED_PIN, LOW);
    lastHeartbeat = millis();
  }

  delay(500);
}

