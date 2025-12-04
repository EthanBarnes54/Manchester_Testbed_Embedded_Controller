// ---------------- Pin control (5 PWM-like + 1 digital) ---------------- //
// Pin Channels:
//  1) squeeze_plate  (PWM 0..1023)
//  2) ion_source     (PWM 0..1023)
//  3) wein_filter    (PWM 0..1023)
//  4) cone_1         (PWM 0..1023)
//  5) cone_2         (PWM 0..1023)
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

namespace {

constexpr int LED_PIN = 2;

constexpr int SQUEEZE_PLATE_PIN = 25;
constexpr int ION_SOURCE_PIN = 26;
constexpr int WEIN_FILTER_PIN = 27;
constexpr int CONE_1_PIN = 32;
constexpr int CONE_2_PIN = 33;
constexpr int SWITCH_LOGIC_PIN = 16;

constexpr int CHANNEL_COUNT = 6;
constexpr int CONTROLLED_PWM_CHANNELS = 5;

constexpr int CHANNEL_PINS[CHANNEL_COUNT] = {
  SQUEEZE_PLATE_PIN, ION_SOURCE_PIN, WEIN_FILTER_PIN, CONE_1_PIN, CONE_2_PIN, SWITCH_LOGIC_PIN
};

constexpr int MODULATION_FREQUENCY = 5000;
constexpr int MODULATION_RESOLUTION = 10;
constexpr int MAX_MODULATION_VALUE = (1 << MODULATION_RESOLUTION) - 1;
constexpr int LED_CONTROL_CHANNELS[CONTROLLED_PWM_CHANNELS] = {0, 1, 2, 3, 4};

constexpr unsigned long MEASUREMENT_INTERVAL_MS = 50;    // periodic ADC read
constexpr unsigned long HEARTBEAT_INTERVAL_MS = 2000;    // LED heartbeat
constexpr int COMMAND_BUFFER_LIMIT = 256;                // serial command size cap

// Allow wider bounds for future flexibility; clamp internally
constexpr int SWITCH_PERIOD_MIN_US = 1;
constexpr int SWITCH_PERIOD_MAX_US = 2000000;            // 2 seconds max

// WiFi/OTA configuration (replace with secure credentials / provisioning)
constexpr const char* WIFI_SSID = "YourSSID";
constexpr const char* WIFI_PASSWORD = "YourPassword";
constexpr const char* OTA_HOSTNAME = "esp32dev";
constexpr int WIFI_CONNECT_ATTEMPTS = 40;  // ~20s at 500ms per attempt

}  // namespace

Adafruit_ADS1115 ads;

class ChannelController {
 public:
  ChannelController() = default;

  void begin() {
    instance_ = this;
    init_pwm_channels();
    configure_switch_timer();
  }

  void set_channel(uint8_t channel_number, int value) {
    if (channel_number < 1 || channel_number > CHANNEL_COUNT) {
      Serial.println("ERROR: PIN index out of range!");
      return;
    }

    const uint8_t channel_index = channel_number - 1;

    if (channel_index < CONTROLLED_PWM_CHANNELS) {
      if (value < 0) value = 0;
      if (value > MAX_MODULATION_VALUE) value = MAX_MODULATION_VALUE;

      ledcWrite(LED_CONTROL_CHANNELS[channel_index], value);
      channel_values_[channel_index] = value;
      Serial.println(String("ACK PIN ") + channel_number + " " + value);
      return;
    }

    stop_switching(value ? 1 : 0);
    Serial.println(String("ACK PIN ") + channel_number + " " + channel_values_[channel_index]);
  }

  bool apply_target_voltages(const String& args) {
    float targets[CONTROLLED_PWM_CHANNELS];

    if (!parse_voltage_targets(args, targets, CONTROLLED_PWM_CHANNELS)) {
      return false;
    }

    for (int i = 0; i < CONTROLLED_PWM_CHANNELS; ++i) {
      const int pwm_value = convert_voltage_to_pwm(targets[i]);
      set_channel(static_cast<uint8_t>(i + 1), pwm_value);
    }

    return true;
  }

  void report_channels() const {
    int snapshot[CHANNEL_COUNT];
    snapshot_values(snapshot, CHANNEL_COUNT);

    Serial.print("PINS ");
    Serial.print("squeeze_plate="); Serial.print(snapshot[0]); Serial.print(" ");
    Serial.print("ion_source=");    Serial.print(snapshot[1]); Serial.print(" ");
    Serial.print("wein_filter=");   Serial.print(snapshot[2]); Serial.print(" ");
    Serial.print("cone_1=");        Serial.print(snapshot[3]); Serial.print(" ");
    Serial.print("cone_2=");        Serial.print(snapshot[4]); Serial.print(" ");
    Serial.print("switch_logic=");  Serial.print(snapshot[5]);
    Serial.println("");
  }

  bool automate_switching(unsigned long period_us) {
    if (switch_timer_ == nullptr) {
      return false;
    }

    portENTER_CRITICAL(&switch_mux_);
    switch_period_us_ = period_us;
    switch_auto_enabled_ = true;

    timerAlarmWrite(switch_timer_, switch_period_us_, true);
    timerAlarmEnable(switch_timer_);

    portEXIT_CRITICAL(&switch_mux_);
    return true;
  }

  void stop_switching(int switch_level) {
    portENTER_CRITICAL(&switch_mux_);
    switch_auto_enabled_ = false;
    switch_period_us_ = 0;

    if (switch_timer_ != nullptr) {
      timerAlarmDisable(switch_timer_);
    }

    channel_values_[5] = switch_level ? 1 : 0;
    portEXIT_CRITICAL(&switch_mux_);
    set_switch_hardware(switch_level != 0);
  }

  void snapshot_values(int* destination, size_t count) const {
    if (destination == nullptr || count < CHANNEL_COUNT) {
      return;
    }

    portENTER_CRITICAL(const_cast<portMUX_TYPE*>(&switch_mux_));
    memcpy(destination, channel_values_, sizeof(channel_values_));
    portEXIT_CRITICAL(const_cast<portMUX_TYPE*>(&switch_mux_));
  }

 private:
  static ChannelController* instance_;

  int channel_values_[CHANNEL_COUNT] = {0};
  hw_timer_t* switch_timer_ = nullptr;
  portMUX_TYPE switch_mux_ = portMUX_INITIALIZER_UNLOCKED;
  bool switch_auto_enabled_ = false;
  unsigned long switch_period_us_ = 0;

  static void IRAM_ATTR on_switch_timer() {
    if (instance_ != nullptr) {
      instance_->toggle_switch();
    }
  }

  void IRAM_ATTR toggle_switch() {
    portENTER_CRITICAL_ISR(&switch_mux_);
    const bool currently_high = (channel_values_[5] != 0);
    const bool next_state = !currently_high;

    set_switch_hardware(next_state);
    channel_values_[5] = next_state ? 1 : 0;
    portEXIT_CRITICAL_ISR(&switch_mux_);
  }

  void set_switch_hardware(bool high) {
    const uint32_t mask = (1UL << SWITCH_LOGIC_PIN);

    if (high) {
      GPIO.out_w1ts = mask;
    } else {
      GPIO.out_w1tc = mask;
    }
  }

  void init_pwm_channels() {
    for (int i = 0; i < CONTROLLED_PWM_CHANNELS; ++i) {
      pinMode(CHANNEL_PINS[i], OUTPUT);
      ledcSetup(LED_CONTROL_CHANNELS[i], MODULATION_FREQUENCY, MODULATION_RESOLUTION);
      ledcAttachPin(CHANNEL_PINS[i], LED_CONTROL_CHANNELS[i]);
      ledcWrite(LED_CONTROL_CHANNELS[i], 0);
      channel_values_[i] = 0;
    }

    pinMode(CHANNEL_PINS[5], OUTPUT);
    set_switch_hardware(false);
    channel_values_[5] = 0;
  }

  void configure_switch_timer() {
    switch_timer_ = timerBegin(0, 80, true);

    if (switch_timer_ != nullptr) {
      timerAttachInterrupt(switch_timer_, &ChannelController::on_switch_timer, true);
      timerAlarmDisable(switch_timer_);
      LOG_INFO("Switch logic timer initialised...");
    } else {
      LOG_ERROR("ERROR: Switch unavailable! (Timer allocation failed)");
    }
  }

  static int convert_voltage_to_pwm(float volts) {
    float clamped = constrain(volts, 0.0f, 3.3f);
    float scaled = (clamped / 3.3f) * static_cast<float>(MAX_MODULATION_VALUE);
    int pwm_value = static_cast<int>(scaled + 0.5f);

    if (pwm_value < 0) return 0;
    if (pwm_value > MAX_MODULATION_VALUE) return MAX_MODULATION_VALUE;
    return pwm_value;
  }

  static bool parse_voltage_targets(String args, float* target_buffer, size_t expected_values) {
    if (target_buffer == nullptr || expected_values == 0) {
      return false;
    }

    args.trim();
    args.replace(',', ' ');

    size_t parsed = 0;
    int start_index = 0;

    while (parsed < expected_values && start_index < args.length()) {
      const int separator = args.indexOf(' ', start_index);
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
};

ChannelController* ChannelController::instance_ = nullptr;

class MeasurementService {
 public:
  explicit MeasurementService(Adafruit_ADS1115& adc) : adc_(adc) {}

  bool begin() {
    return adc_.begin();
  }

  float read_voltage() {
    int16_t raw_voltage = adc_.readADC_SingleEnded(0);
    return adc_.computeVolts(raw_voltage);
  }

  void maybe_sample(unsigned long now_ms) {
    if (now_ms - last_measurement_ms_ < MEASUREMENT_INTERVAL_MS) {
      return;
    }

    const float measured_voltage = read_voltage();
    Serial.println(String("MEASURED ") + measured_voltage + " V");
    last_measurement_ms_ = now_ms;
  }

 private:
  Adafruit_ADS1115& adc_;
  unsigned long last_measurement_ms_ = 0;
};

class LedIndicator {
 public:
  explicit LedIndicator(int pin) : pin_(pin) {}

  void begin() {
    pinMode(pin_, OUTPUT);
    digitalWrite(pin_, HIGH);
  }

  void heartbeat(unsigned long now_ms) {
    if (now_ms - last_heartbeat_ms_ > HEARTBEAT_INTERVAL_MS) {
      digitalWrite(pin_, HIGH);
      delay(50);
      digitalWrite(pin_, LOW);
      last_heartbeat_ms_ = now_ms;
    }
  }

  void blink_once(int duration_ms = 100) {
    digitalWrite(pin_, HIGH);
    delay(duration_ms);
    digitalWrite(pin_, LOW);
  }

  void blink_error(int flash_count = 3, int duration_ms = 100) {
    for (int i = 0; i < flash_count; ++i) {
      digitalWrite(pin_, HIGH);
      delay(duration_ms);
      digitalWrite(pin_, LOW);
      delay(duration_ms);
    }
  }

  void set_low() {
    digitalWrite(pin_, LOW);
  }

 private:
  int pin_;
  unsigned long last_heartbeat_ms_ = 0;
};

class OtaWifiService {
 public:
  void begin(const char* ssid, const char* password, const char* hostname) {
    connect(ssid, password);

    ArduinoOTA.setHostname(hostname);

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

    if (WiFi.status() == WL_CONNECTED) {
      ArduinoOTA.begin();
      ota_enabled_ = true;
      Serial.println("Ready for OTA updates...");
    }
  }

  void loop() {
    if (WiFi.status() != WL_CONNECTED) {
      reconnect();
      return;
    }

    if (ota_enabled_) {
      ArduinoOTA.handle();
    }
  }

 private:
  bool ota_enabled_ = false;
  unsigned long last_reconnect_attempt_ms_ = 0;

  void connect(const char* ssid, const char* password) {
    WiFi.begin(ssid, password);

    int attempts = 0;
    while (WiFi.status() != WL_CONNECTED && attempts < WIFI_CONNECT_ATTEMPTS) {
      delay(500);
      Serial.print(".");
      ++attempts;
    }

    if (WiFi.status() == WL_CONNECTED) {
      Serial.println("\nWiFi connected, IP address:");
      Serial.println(WiFi.localIP());
    } else {
      Serial.println("\nWARNING: WiFi connection failed, running without OTA...");
    }
  }

  void reconnect() {
    const unsigned long now = millis();
    if (now - last_reconnect_attempt_ms_ < 2000) {
      return;
    }

    last_reconnect_attempt_ms_ = now;
    Serial.println("WiFi lost, attempting reconnection...");
    connect(WIFI_SSID, WIFI_PASSWORD);

    if (WiFi.status() == WL_CONNECTED && !ota_enabled_) {
      ArduinoOTA.begin();
      ota_enabled_ = true;
    }
  }
};

class CommandProcessor {
 public:
  CommandProcessor(ChannelController& channels, MeasurementService& measurement, LedIndicator& leds)
      : channels_(channels), measurement_(measurement), leds_(leds) {}

  void poll_serial() {
    while (Serial.available()) {
      char incoming_char = static_cast<char>(Serial.read());

      if (incoming_char == '\r') {
        continue;
      }

      if (incoming_char == '\n') {
        handle_command(command_buffer_);
        command_buffer_ = "";
      } else {
        if (command_buffer_.length() < COMMAND_BUFFER_LIMIT) {
          command_buffer_ += incoming_char;
        } else {
          command_buffer_ = "";
        }
      }
    }
  }

 private:
  ChannelController& channels_;
  MeasurementService& measurement_;
  LedIndicator& leds_;
  String command_buffer_;

  static uint8_t to_pin_index(String token) {
    token.trim();
    String token_lower = token;
    token_lower.toLowerCase();

    if (token_lower == "squeeze_plate") return 1;
    if (token_lower == "ion_source") return 2;
    if (token_lower == "wein_filter") return 3;
    if (token_lower == "cone_1") return 4;
    if (token_lower == "cone_2") return 5;
    if (token_lower == "switch_logic") return 6;
    return static_cast<uint8_t>(token.toInt());
  }

  void handle_command(String command) {
    command.trim();

    if (command.isEmpty()) {
      return;
    }

    if (command.startsWith("TARGETS")) {
      const int first_space_index = command.indexOf(' ');

      if (first_space_index <= 0) {
        Serial.println("ERROR: TARGETS requires five voltages!");
        return;
      }

      const String args = command.substring(first_space_index + 1);

      if (channels_.apply_target_voltages(args)) {
        Serial.println("ACK TARGETS");
      } else {
        Serial.println("ERROR: TARGETS requires five voltages!");
      }
    } else if (command.startsWith("PIN")) {
      const int first_space_index = command.indexOf(' ');
      const int second_space_index = command.indexOf(' ', first_space_index + 1);

      if (first_space_index > 0 && second_space_index > first_space_index) {
        const String token = command.substring(first_space_index + 1, second_space_index);
        const uint8_t pin_index = to_pin_index(token);

        const int value = command.substring(second_space_index + 1).toInt();
        channels_.set_channel(pin_index, value);
      } else {
        Serial.println("ERROR: Invalid PIN syntax!");
      }
    } else if (command.startsWith("SWITCH_PERIOD_US")) {
      const int first_space_index = command.indexOf(' ');

      if (first_space_index <= 0) {
        Serial.println("ERROR: Invalid switch time!");
        return;
      }

      String value_token = command.substring(first_space_index + 1);
      value_token.trim();
      long switch_period_us = value_token.toInt();

      if (switch_period_us <= 0) {
        channels_.stop_switching(0);
        Serial.println("ACK SWITCH_PERIOD_US 0 (disabled)");
        return;
      }

      if (switch_period_us < SWITCH_PERIOD_MIN_US || switch_period_us > SWITCH_PERIOD_MAX_US) {
        Serial.println("ERROR: Switch time out of bounds!");
        return;
      }

      if (!channels_.automate_switching(static_cast<unsigned long>(switch_period_us))) {
        Serial.println("ERROR: Switch timer unavailable!");
        return;
      }

      Serial.print("ACK SWITCH_PERIOD_US ");
      Serial.println(switch_period_us);
    } else if (command.equalsIgnoreCase("GET PINS") || command.equalsIgnoreCase("PINS")) {
      channels_.report_channels();
    } else if (command.equalsIgnoreCase("READ")) {
      const float live_voltage = measurement_.read_voltage();
      Serial.println(String("MEASURED ") + live_voltage + " V");
      leds_.blink_once(80);
    } else if (command.equalsIgnoreCase("PING")) {
      Serial.println("OK");
      leds_.blink_once(50);
    } else {
      Serial.println("ERROR: Unknown command received!");
      leds_.blink_error(2, 70);
    }
  }
};

ChannelController channels;
MeasurementService measurement_service(ads);
LedIndicator led_indicator(LED_PIN);
OtaWifiService ota_wifi_service;
CommandProcessor command_processor(channels, measurement_service, led_indicator);

void setup() {
  led_indicator.begin();

  Serial.begin(115200);
  delay(1500);

  LOG_INFO("System initialising...");

  Wire.begin();

  if (!measurement_service.begin()) {
    LOG_ERROR("ERROR: ADS1115 not found!");
    led_indicator.blink_error(5);
  } else {
    LOG_INFO("ADS1115 ready...");
  }

  channels.begin();
  ota_wifi_service.begin(WIFI_SSID, WIFI_PASSWORD, OTA_HOSTNAME);

  LOG_INFO("Setup complete. Awaiting commands...");
  led_indicator.set_low();
}

void loop() {
  command_processor.poll_serial();

  const unsigned long now_ms = millis();
  measurement_service.maybe_sample(now_ms);
  led_indicator.heartbeat(now_ms);
  ota_wifi_service.loop();

  delay(1);
}
