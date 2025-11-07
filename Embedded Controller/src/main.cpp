#include <Arduino.h>
#include "log.h"
#include <Wire.h>
#include <ESP8266WiFi.h>
#include <ArduinoOTA.h>
#include <Adafruit_ADS1X15.h>
#include <Adafruit_MCP4725.h>

Adafruit_ADS1115 ads;
Adafruit_MCP4725 dac;

const int LED_PIN = 2;

float TargetVoltage = 0.0;
unsigned long lastHeartbeat = 0;

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

// ---------- Setup ---------- //

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
  
  ArduinoOTA.setHostname("esp12fdev");

  ArduinoOTA.onStart([]() {
    Serial.println("OTA connection starting...");
  });

  ArduinoOTA.onEnd([]() {
    Serial.println("OTA connection established...");
  });

  ArduinoOTA.onProgress([](unsigned int progress, unsigned int total) {
    Serial.printf("Connection in Progress: %u%%\r", (progress / (total / 100)));
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
  Serial.println("Ready for OTA updates");

  // --- I2C & Peripheral Setup ---
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
}

// ---------- Voltage control ---------- //

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

// ---------- Main loop ---------- //

void loop() {
  
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();

    if (cmd.startsWith("SET")) {
      float voltage_target = cmd.substring(4).toFloat();
      SetVoltage(voltage_target);
      Serial.println("ACK SET");
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



// THESE ARE EXAMPLES FOR TESTING, CHANGE ONCE RIG IS FINISHED
// Read a voltage from ADC channel 0
// int16_t adc0 = ads.readADC_SingleEnded(0);
// float voltage = adc0 * 0.1875 / 1000.0;  // convert to volts

// Set DAC output proportional to read voltage
// uint16_t dacValue = (uint16_t)((voltage / 3.3) * 4095);
// dac.setVoltage(dacValue, false);

// LOG_DEBUG(String("ADC0: ") + voltage + " V | DAC: " + dacValue);
