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
#include <cstring>
#include <WiFi.h>
#include <ESPmDNS.h>
#include <WiFiUdp.h>
#include <ArduinoOTA.h>
#include <Adafruit_ADS1X15.h>
#include <Adafruit_MCP4725.h>
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "soc/gpio_struct.h"

Adafruit_ADS1115 ads;
Adafruit_MCP4725 dac;

const int LED_PIN = 2; 

float target_voltage = 0.0;
unsigned long last_heartbeat_ms = 0;
unsigned long last_measurement_ms = 0;

// Automatic switching for logic output (channel 6)
bool switch_auto_enabled = false;
unsigned long switch_period_us = 0;
hw_timer_t* switch_timer = nullptr;
portMUX_TYPE switch_mux = portMUX_INITIALIZER_UNLOCKED;

String command_buffer;

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
int channel_values[6] = {0, 0, 0, 0, 0, 0};

static const int PWM_FREQ = 5000;         
static const int PWM_RESOLUTION = 10;     
static const int PWM_MAX = (1 << PWM_RESOLUTION) - 1;
static const int LEDC_CHANNELS[5] = {0, 1, 2, 3, 4};

static inline void fast_set_switch_level(bool high) {
  const uint32_t mask = (1UL << SWITCH_LOGIC_PIN);
  if (high) {
    GPIO.out_w1ts = mask;
  } else {
    GPIO.out_w1tc = mask;
  }
}

void IRAM_ATTR on_switch_timer() {
  portENTER_CRITICAL_ISR(&switch_mux);
  const bool currently_high = (channel_values[5] != 0);
  const bool next_state = !currently_high;
  fast_set_switch_level(next_state);
  channel_values[5] = next_state ? 1 : 0;
  portEXIT_CRITICAL_ISR(&switch_mux);
}

bool configure_switch_automation(unsigned long period_us) {
  if (switch_timer == nullptr) {
    return false;
  }
  portENTER_CRITICAL(&switch_mux);
  switch_period_us = period_us;
  switch_auto_enabled = true;
  timerAlarmWrite(switch_timer, switch_period_us, true);
  timerAlarmEnable(switch_timer);
  portEXIT_CRITICAL(&switch_mux);
  return true;
}

void stop_switch_automation_and_set_level(int switch_level) {
  portENTER_CRITICAL(&switch_mux);
  switch_auto_enabled = false;
  switch_period_us = 0;
  if (switch_timer != nullptr) {
    timerAlarmDisable(switch_timer);
  }
  channel_values[5] = switch_level ? 1 : 0;
  portEXIT_CRITICAL(&switch_mux);
  fast_set_switch_level(switch_level != 0);
}

void init_channels() {
  
  for (int i = 0; i < 5; i++) {
    pinMode(CHANNEL_PINS[i], OUTPUT);
    ledcSetup(LEDC_CHANNELS[i], PWM_FREQ, PWM_RESOLUTION);
    ledcAttachPin(CHANNEL_PINS[i], LEDC_CHANNELS[i]);
    ledcWrite(LEDC_CHANNELS[i], 0);
    channel_values[i] = 0;
  }
  
  pinMode(CHANNEL_PINS[5], OUTPUT);
  fast_set_switch_level(false);
  channel_values[5] = 0;
}

void set_channel(uint8_t channel_index_1based, int value) {
  if (channel_index_1based < 1 || channel_index_1based > 6) {
    Serial.println("ERROR: PIN index out of range!");
    return;
  }
  uint8_t channel_index = channel_index_1based - 1;
  int pin_number = CHANNEL_PINS[channel_index];
  if (channel_index < 5) {
    if (value < 0) value = 0;
    if (value > PWM_MAX) value = PWM_MAX;
    ledcWrite(LEDC_CHANNELS[channel_index], value);
    channel_values[channel_index] = value;
    Serial.println(String("ACK PIN ") + channel_index_1based + " " + value);
  } else {
    stop_switch_automation_and_set_level(value ? 1 : 0);
    Serial.println(String("ACK PIN ") + channel_index_1based + " " + channel_values[channel_index]);
  }
}

void report_channels() {
  int snapshot_values[6];
  portENTER_CRITICAL(&switch_mux);
  memcpy(snapshot_values, channel_values, sizeof(snapshot_values));
  portEXIT_CRITICAL(&switch_mux);

  Serial.print("PINS ");
  Serial.print("squeeze_plate="); Serial.print(snapshot_values[0]); Serial.print(" ");
  Serial.print("ion_source=");    Serial.print(snapshot_values[1]); Serial.print(" ");
  Serial.print("wein_filter=");   Serial.print(snapshot_values[2]); Serial.print(" ");
  Serial.print("cone_1=");        Serial.print(snapshot_values[3]); Serial.print(" ");
  Serial.print("cone_2=");        Serial.print(snapshot_values[4]); Serial.print(" ");
  Serial.print("switch_logic=");  Serial.print(snapshot_values[5]);
  Serial.println("");
}

void blink_once(int duration_ms = 100) {
  digitalWrite(LED_PIN, HIGH);
  delay(duration_ms);
  digitalWrite(LED_PIN, LOW);
}

void blink_error(int flash_count = 3, int duration_ms = 100) {
  for (int i = 0; i < flash_count; i++) {
    digitalWrite(LED_PIN, HIGH);
    delay(duration_ms);
    digitalWrite(LED_PIN, LOW);
    delay(duration_ms);
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
    blink_error(5);
  } else {
    LOG_INFO("ADS1115 ready...");
  }

  if (!dac.begin(0x60)) {
    LOG_ERROR("MCP4725 not found!");
    blink_error(5);
  } else {
    LOG_INFO("MCP4725 ready...");
  }

  LOG_INFO("Setup complete. Awaiting commands...");
  digitalWrite(LED_PIN, LOW);

  init_channels();

  switch_timer = timerBegin(0, 80, true);
  if (switch_timer != nullptr) {
    timerAttachInterrupt(switch_timer, &on_switch_timer, true);
    timerAlarmDisable(switch_timer);
    LOG_INFO("Switch logic timer initialised...");
  } else {
    LOG_ERROR("Failed to allocate switch logic timer. Automatic switching disabled.");
  }
}

void set_voltage(float volts) {
  if (volts < 0) volts = 0;
  if (volts > 3.3) volts = 3.3;

  uint16_t dacValue = (uint16_t)((volts / 3.3) * 4095.0);
  dac.setVoltage(dacValue, false);
  target_voltage = volts;

  Serial.println(String("DAC set to ") + volts + " V");
  blink_once();  
}

float read_voltage() {
  int16_t raw_voltage = ads.readADC_SingleEnded(0);
  return ads.computeVolts(raw_voltage);
}

void handle_command(String command) {
  command.trim();
  if (command.isEmpty()) {
    return;
  }

  if (command.startsWith("SET")) {
    float voltage_target = command.substring(4).toFloat();
    set_voltage(voltage_target);
    Serial.println("ACK SET");
  }
  else if (command.startsWith("PIN")) {
    int first_space_index = command.indexOf(' ');
    int second_space_index = command.indexOf(' ', first_space_index + 1);
    if (first_space_index > 0 && second_space_index > first_space_index) {
      String token = command.substring(first_space_index + 1, second_space_index);
      token.trim();
      uint8_t pin_index = 0;
      String token_lower = token;
      token_lower.toLowerCase();

      if (token_lower == "squeeze_plate") pin_index = 1;
      else if (token_lower == "ion_source") pin_index = 2;
      else if (token_lower == "wein_filter") pin_index = 3;
      else if (token_lower == "cone_1") pin_index = 4;
      else if (token_lower == "cone_2") pin_index = 5;
      else if (token_lower == "switch_logic") pin_index = 6;
      else pin_index = (uint8_t)token.toInt();

      int value = command.substring(second_space_index + 1).toInt();
      set_channel(pin_index, value);
    } else {
      Serial.println("ERROR: PIN syntax");
    }
  }
  else if (command.startsWith("SWITCH_PERIOD_US")) {
    int first_space_index = command.indexOf(' ');
    if (first_space_index <= 0) {
      Serial.println("ERROR: SWITCH_PERIOD_US syntax");
      return;
    }
    String value_token = command.substring(first_space_index + 1);
    value_token.trim();
    long switch_period_us = value_token.toInt();
    if (switch_period_us <= 0) {
      stop_switch_automation_and_set_level(0);
      Serial.println("ACK SWITCH_PERIOD_US 0 (disabled)");
      return;
    }
    if (switch_period_us < 1 || switch_period_us > 20) {
      Serial.println("ERROR: SWITCH_PERIOD_US must be between 1 and 20 microseconds");
      return;
    }
    if (!configure_switch_automation(static_cast<unsigned long>(switch_period_us))) {
      Serial.println("ERROR: Switch timer unavailable");
      return;
    }
    Serial.print("ACK SWITCH_PERIOD_US ");
    Serial.println(switch_period_us);
  }
  else if (command.equalsIgnoreCase("GET PINS") || command.equalsIgnoreCase("PINS")) {
    report_channels();
  }
  else if (command.equalsIgnoreCase("READ")) {
    float live_voltage = read_voltage();
    Serial.println(String("MEASURED ") + live_voltage + " V");
    blink_once(80);
  }
  else if (command.equalsIgnoreCase("PING")) {
    Serial.println("OK");
    blink_once(50);
  }
  else {
    Serial.println("ERROR: Unknown command");
    blink_error(2, 70);
  }
}

void loop() {
  while (Serial.available()) {
    char incoming_char = static_cast<char>(Serial.read());
    if (incoming_char == '\r') {
      continue;
    }
    if (incoming_char == '\n') {
      handle_command(command_buffer);
      command_buffer = "";
    } else {
      if (command_buffer.length() < 256) {
        command_buffer += incoming_char;
      } else {command_buffer = "";}
    }
  }

  unsigned long now_ms = millis();

  if (now_ms - last_measurement_ms >= 50) {  
    float measured_voltage = read_voltage();
    Serial.println(String("MEASURED ") + measured_voltage + " V");
    last_measurement_ms = now_ms;
  }

  if (now_ms - last_heartbeat_ms > 2000) {
    digitalWrite(LED_PIN, HIGH);
    delay(50);
    digitalWrite(LED_PIN, LOW);
    last_heartbeat_ms = now_ms;
  }

  delay(1); 
}
