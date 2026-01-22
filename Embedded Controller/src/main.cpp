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
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "soc/gpio_struct.h"

Adafruit_ADS1115 ads;

const int LED_PIN = 2; 

unsigned long last_heartbeat_ms = 0;
unsigned long last_measurement_ms = 0;

bool switch_auto_enabled = false;
unsigned long switch_period_us = 0;

hw_timer_t* switch_timer = nullptr;
portMUX_TYPE switch_mux = portMUX_INITIALIZER_UNLOCKED;

String command_buffer;

#define SQUEEZE_PLATE_PIN 25  
#define ION_SOURCE_PIN 26     
#define WEIN_FILTER_PIN 27   
#define CONE_1_PIN 32        
#define CONE_2_PIN 33       
#define SWITCH_LOGIC_PIN 16   


const int CHANNEL_PINS[6] = {SQUEEZE_PLATE_PIN, ION_SOURCE_PIN, WEIN_FILTER_PIN, CONE_1_PIN, CONE_2_PIN, SWITCH_LOGIC_PIN};
int channel_values[6] = {0, 0, 0, 0, 0, 0};

static const int MODULATION_FREQUENCY = 5000;         
static const int MODULATION_RESOLUTION = 10;     
static const int MAX_MODULATION_VALUE = (1 << MODULATION_RESOLUTION) - 1;
static const int LED_CONTROL_CHANNELS[5] = {0, 1, 2, 3, 4};

static inline void set_switch(bool high) {
  const uint32_t mask = (1UL << SWITCH_LOGIC_PIN);

  if (high) {
    GPIO.out_w1ts = mask;
  } else {
    GPIO.out_w1tc = mask;
  }
}

void IRAM_ATTR switch_timings() {

  portENTER_CRITICAL_ISR(&switch_mux);
  const bool currently_high = (channel_values[5] != 0);
  const bool next_state = !currently_high;

  set_switch(next_state);

  channel_values[5] = next_state ? 1 : 0;
  portEXIT_CRITICAL_ISR(&switch_mux);
}

bool automate_switching(unsigned long period_us) {

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

void stop_switching(int switch_level) {
  portENTER_CRITICAL(&switch_mux);
  switch_auto_enabled = false;
  switch_period_us = 0;

  if (switch_timer != nullptr) {
    timerAlarmDisable(switch_timer);
  }

  channel_values[5] = switch_level ? 1 : 0;
  portEXIT_CRITICAL(&switch_mux);
  set_switch(switch_level != 0);
}

void init_channels() {
  
  for (int i = 0; i < 5; i++) {
    pinMode(CHANNEL_PINS[i], OUTPUT);
    ledcSetup(LED_CONTROL_CHANNELS[i], MODULATION_FREQUENCY, MODULATION_RESOLUTION);
    ledcAttachPin(CHANNEL_PINS[i], LED_CONTROL_CHANNELS[i]);
    ledcWrite(LED_CONTROL_CHANNELS[i], 0);

    channel_values[i] = 0;
  }
  
  pinMode(CHANNEL_PINS[5], OUTPUT);
  set_switch(false);
  channel_values[5] = 0;
}

void set_channel(uint8_t channel_number, int value) {
  if (channel_number < 1 || channel_number > 6) {
    Serial.println("ERROR: PIN index out of range!");
    return;
  }

  uint8_t channel_index = channel_number - 1;
  int pin_number = CHANNEL_PINS[channel_index];

  if (channel_index < 5) {
    if (value < 0) value = 0;
    if (value > MAX_MODULATION_VALUE) value = MAX_MODULATION_VALUE;

    ledcWrite(LED_CONTROL_CHANNELS[channel_index], value);
    channel_values[channel_index] = value;
    Serial.println(String("ACK PIN ") + channel_number + " " + value);
  } else {
    stop_switching(value ? 1 : 0);
    Serial.println(String("ACK PIN ") + channel_number + " " + channel_values[channel_index]);
  }
}

int convert_voltage_to_pwm(float volts) {
  float clamped = constrain(volts, 0.0f, 3.3f);
  float scaled = (clamped / 3.3f) * static_cast<float>(MAX_MODULATION_VALUE);
  int pwm_value = static_cast<int>(scaled + 0.5f);
  if (pwm_value < 0) {
    return 0;
  }
  if (pwm_value > MAX_MODULATION_VALUE) {
    return MAX_MODULATION_VALUE;
  }
  return pwm_value;
}

bool parse_voltage_targets(String args, float* target_buffer, size_t expected_values) {
  if (target_buffer == nullptr || expected_values == 0) {
    return false;
  }

  args.trim();
  args.replace(',', ' ');

  size_t parsed = 0;
  int start_index = 0;

  while (parsed < expected_values && start_index < args.length()) {
    int separator = args.indexOf(' ', start_index);
    String token;

    if (separator == -1) {
      token = args.substring(start_index);
      start_index = args.length();
    } else {
      token = args.substring(start_index, separator);
      start_index = separator + 1;
    }

    token.trim();
    if (token.length() == 0) {
      continue;
    }

    target_buffer[parsed++] = token.toFloat();
  }

  return parsed == expected_values;
}

bool apply_target_voltages(const String& args) {
  float targets[5];

  if (!parse_voltage_targets(args, targets, 5)) {
    return false;
  }

  for (int i = 0; i < 5; ++i) {
    int pwm_value = convert_voltage_to_pwm(targets[i]);
    set_channel(static_cast<uint8_t>(i + 1), pwm_value);
  }

  return true;
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

  ArduinoOTA.onProgress([](unsigned int connection_progress, unsigned int connection_capacity) {
    if (connection_capacity) {
      Serial.printf("OTA Progress: %u%%\r", (connection_progress * 100) / connection_capacity);
    }
  });

  ArduinoOTA.onError([](ota_error_t error) {
    Serial.printf("Error[%u]: ", error);

    if (error == OTA_AUTH_ERROR) Serial.println("ERROR: Authentication Failed!");
    else if (error == OTA_BEGIN_ERROR) Serial.println("ERROR: Begin Failed!"); 
    else if (error == OTA_CONNECT_ERROR) Serial.println("ERROR: Connect Failed!");
    else if (error == OTA_RECEIVE_ERROR) Serial.println("ERROR: Receive Failed!");
    else if (error == OTA_END_ERROR) Serial.println("ERROR: End Failed!");
  });

  ArduinoOTA.begin();
  Serial.println("Ready for OTA updates...");

  Wire.begin();

  if (!ads.begin()) {
    LOG_ERROR("ERROR: ADS1115 not found!");
    blink_error(5);

  } else {
    LOG_INFO("ADS1115 ready...");
  }

  LOG_INFO("Setup complete. Awaiting commands...");
  digitalWrite(LED_PIN, LOW);

  init_channels();

  switch_timer = timerBegin(0, 80, true);

  if (switch_timer != nullptr) {
    timerAttachInterrupt(switch_timer, &switch_timings, true);
    timerAlarmDisable(switch_timer);
    LOG_INFO("Switch logic timer initialised...");
  } else {
    LOG_ERROR("ERROR: Switch unavailable! (Timer allocation failed)");
  }
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

  if (command.startsWith("TARGETS")) {
    int first_space_index = command.indexOf(' ');

    if (first_space_index <= 0) {
      Serial.println("ERROR: TARGETS requires five voltages!");
      return;
    }

    String args = command.substring(first_space_index + 1);

    if (apply_target_voltages(args)) {
      Serial.println("ACK TARGETS");
    } else {
      Serial.println("ERROR: TARGETS requires five voltages!");
    }
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
      Serial.println("ERROR: Invalid PIN syntax!");
    }
  }

  else if (command.startsWith("SWITCH_PERIOD_US")) {
    int first_space_index = command.indexOf(' ');

    if (first_space_index <= 0) {
      Serial.println("ERROR: Invalid switch time!");
      return;
    }

    String value_token = command.substring(first_space_index + 1);
    value_token.trim();
    long switch_period_us = value_token.toInt();

    if (switch_period_us <= 0) {
      stop_switching(0);
      Serial.println("ACK SWITCH_PERIOD_US 0 (disabled)");
      return;
    }

    if (switch_period_us < 1 || switch_period_us > 20) {
      Serial.println("ERROR: Switch time out of bounds!");
      return;
    }

    if (!automate_switching(static_cast<unsigned long>(switch_period_us))) {
      Serial.println("ERROR: Switch timer unavailable!");
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
    Serial.println("ERROR: Unknown command received!");
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
