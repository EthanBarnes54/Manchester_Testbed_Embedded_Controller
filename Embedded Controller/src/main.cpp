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

namespace{

constexpr int LED_PIN = 2;

constexpr int SQUEEZE_PLATE_PIN = 25;
constexpr int ION_SOURCE_PIN = 26;
constexpr int WEIN_FILTER_PIN = 27;
constexpr int CONE_1_PIN = 32;
constexpr int CONE_2_PIN = 33;
constexpr int SWITCH_LOGIC_PIN = 16;

constexpr int CHANNEL_COUNT = 6;
constexpr int CONTROLLED_PULSE_CHANNELS = 5;

constexpr int CHANNEL_PINS[CHANNEL_COUNT] = {SQUEEZE_PLATE_PIN, ION_SOURCE_PIN, WEIN_FILTER_PIN, CONE_1_PIN, CONE_2_PIN, SWITCH_LOGIC_PIN};

constexpr int MODULATION_FREQUENCY = 5000;
constexpr int MODULATION_RESOLUTION = 10;
constexpr int MAX_MODULATION_VALUE = (1 << MODULATION_RESOLUTION) - 1;
constexpr int LED_CONTROL_CHANNELS[CONTROLLED_PULSE_CHANNELS] = {0, 1, 2, 3, 4};

constexpr unsigned long MEASUREMENT_INTERVAL_MS = 50;

// +/-4.096 V is the tightest ADS1115 range that still covers the 0-3.3 V diode signal,
// giving 125 uV per count against 187.5 uV at the library's +/-6.144 V default.
constexpr adsGain_t ADC_GAIN = GAIN_ONE;

// Printed resolution has to sit below one ADC count or the serial link throws the
// precision away. String(float) defaults to 2 places, which is 10 mV.
constexpr unsigned int MEASURED_DECIMAL_PLACES = 5;
constexpr unsigned long HEARTBEAT_INTERVAL_MS = 2000;
constexpr int COMMAND_BUFFER_LIMIT = 256;

// Host silence tolerated before the outputs are dropped. The backend keepalive runs
// well inside this, so only a genuinely dead host trips it.
constexpr unsigned long COMMAND_TIMEOUT_MS = 5000;

// Floor is set by what the timer ISR can actually service - entry, the critical
// section and the GPIO write cost a few microseconds on their own, so anything
// faster than this starves the main loop rather than switching cleanly.
constexpr int SWITCH_PERIOD_MIN_US = 50;
constexpr int SWITCH_PERIOD_MAX_US = 2000000;

// WiFi/OTA configuration. Override these from platformio.ini build_flags rather than
// editing them here, so site credentials never end up committed:
//   -DTESTBED_WIFI_SSID='"YourNetwork"' -DTESTBED_OTA_PASSWORD='"YourOtaPassword"'

#ifndef TESTBED_WIFI_SSID
#define TESTBED_WIFI_SSID "YourSSID"
#endif

#ifndef TESTBED_WIFI_PASSWORD
#define TESTBED_WIFI_PASSWORD "YourPassword"
#endif

#ifndef TESTBED_OTA_PASSWORD
#define TESTBED_OTA_PASSWORD ""
#endif

constexpr const char* WIFI_SSID = TESTBED_WIFI_SSID;
constexpr const char* WIFI_PASSWORD = TESTBED_WIFI_PASSWORD;
constexpr const char* OTA_PASSWORD = TESTBED_OTA_PASSWORD;
constexpr const char* OTA_HOSTNAME = "esp32dev";

// Placeholder SSID shipped with the repo - WiFi stays down until this is replaced.
constexpr const char* WIFI_SSID_PLACEHOLDER = "YourSSID";

constexpr unsigned long WIFI_ASSOCIATE_TIMEOUT_MS = 15000;
constexpr unsigned long WIFI_RETRY_INTERVAL_MS = 30000;

}

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

    if (channel_index < CONTROLLED_PULSE_CHANNELS) {
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
    float targets[CONTROLLED_PULSE_CHANNELS];

    if (!parse_voltage_targets(args, targets, CONTROLLED_PULSE_CHANNELS)) {
      return false;
    }

    for (int i = 0; i < CONTROLLED_PULSE_CHANNELS; ++i) {
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

  // Drops every output back to zero and stops the switch line. Used when the host
  // stops talking to us, so a latched target cannot outlive the controlling process.
  void engage_safe_state() {
    for (int i = 0; i < CONTROLLED_PULSE_CHANNELS; ++i) {
      ledcWrite(LED_CONTROL_CHANNELS[i], 0);
      channel_values_[i] = 0;
    }

    stop_switching(0);
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
    for (int i = 0; i < CONTROLLED_PULSE_CHANNELS; ++i) {
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
    adc_.setGain(ADC_GAIN);
    return adc_.begin();
  }

  float read_voltage() {
    int16_t raw_voltage = adc_.readADC_SingleEnded(0);
    return adc_.computeVolts(raw_voltage);
  }

  // Single formatter for every MEASURED line, so the streamed and on-demand readings
  // always carry the same precision.
  void report_voltage(float volts) {
    Serial.println(String("MEASURED ") + String(volts, MEASURED_DECIMAL_PLACES) + " V");
  }

  void maybe_sample(unsigned long now_ms) {
    if (now_ms - last_measurement_ms_ < MEASUREMENT_INTERVAL_MS) {
      return;
    }

    report_voltage(read_voltage());
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
  // The radio is driven entirely from loop() - nothing here is allowed to block,
  // because serial commands and ADC sampling share the same thread.
  void begin(const char* ssid, const char* password, const char* hostname) {
    ssid_ = ssid;
    password_ = password;

    if (!credentials_configured()) {
      link_state_ = LinkState::Disabled;
      Serial.println("WARNING: WiFi credentials not set, running on serial only...");
      return;
    }

    ota_allowed_ = strlen(OTA_PASSWORD) > 0;

    if (ota_allowed_) {
      ArduinoOTA.setPassword(OTA_PASSWORD);
    } else {
      Serial.println("WARNING: OTA password not set, over the air updates disabled...");
    }

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

    WiFi.mode(WIFI_STA);
    WiFi.setAutoReconnect(true);
    start_association();
  }

  void loop() {
    switch (link_state_) {
      case LinkState::Disabled:
        return;

      case LinkState::Associating:
        if (WiFi.status() == WL_CONNECTED) {
          enter_online_state();
        } else if (millis() - state_entered_ms_ >= WIFI_ASSOCIATE_TIMEOUT_MS) {
          enter_waiting_state("WARNING: WiFi association timed out, retrying later...");
        }
        return;

      case LinkState::Online:
        if (WiFi.status() != WL_CONNECTED) {
          enter_waiting_state("WARNING: WiFi link lost, OTA suspended...");
          return;
        }

        if (ota_started_) {
          ArduinoOTA.handle();
        }

        return;

      case LinkState::Waiting:
        if (millis() - state_entered_ms_ >= WIFI_RETRY_INTERVAL_MS) {
          WiFi.disconnect();
          start_association();
        }
        return;
    }
  }

  bool is_online() const {
    return link_state_ == LinkState::Online;
  }

 private:
  enum class LinkState { Disabled, Associating, Online, Waiting };

  LinkState link_state_ = LinkState::Disabled;
  unsigned long state_entered_ms_ = 0;
  bool ota_started_ = false;
  bool ota_allowed_ = false;
  const char* ssid_ = nullptr;
  const char* password_ = nullptr;

  bool credentials_configured() const {
    return ssid_ != nullptr && strlen(ssid_) > 0 && strcmp(ssid_, WIFI_SSID_PLACEHOLDER) != 0;
  }

  // Kicks off association and returns straight away - WiFi.begin() does not block.
  void start_association() {
    WiFi.begin(ssid_, password_);
    link_state_ = LinkState::Associating;
    state_entered_ms_ = millis();
    Serial.println("WiFi associating...");
  }

  void enter_online_state() {
    link_state_ = LinkState::Online;
    state_entered_ms_ = millis();

    if (ota_allowed_ && !ota_started_) {
      ArduinoOTA.begin();
      ota_started_ = true;
      Serial.println("Ready for OTA updates...");
    }

    Serial.print("WiFi connected, IP address: ");
    Serial.println(WiFi.localIP());
  }

  // Drops OTA on the way down so the next association rebinds against the new address.
  void enter_waiting_state(const char* reason) {
    if (ota_started_) {
      ArduinoOTA.end();
      ota_started_ = false;
    }

    link_state_ = LinkState::Waiting;
    state_entered_ms_ = millis();
    Serial.println(reason);
  }
};

class CommandProcessor {
 public:
  CommandProcessor(ChannelController& channels, MeasurementService& measurement, LedIndicator& leds)
      : channels_(channels), measurement_(measurement), leds_(leds) {}

  bool has_received_command() const {
    return command_seen_;
  }

  unsigned long last_command_ms() const {
    return last_command_ms_;
  }

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
  unsigned long last_command_ms_ = 0;
  bool command_seen_ = false;

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

    last_command_ms_ = millis();
    command_seen_ = true;

    if (command.equalsIgnoreCase("PING")) {
      Serial.println("OK");
    } else if (command.equalsIgnoreCase("READ")) {
      measurement_.report_voltage(measurement_.read_voltage());
      leds_.blink_once(80);
    } else if (command.equalsIgnoreCase("GET PINS") || command.equalsIgnoreCase("PINS")) {
      channels_.report_channels();
    } else if (command.startsWith("TARGETS")) {
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

bool failsafe_engaged = false;

// Once the host has spoken to us it is expected to keep doing so. If it goes quiet
// the outputs are dropped, otherwise a set of targets would stay latched on the rig
// for as long as the board has power.
void enforce_command_timeout(unsigned long now_ms) {
  if (!command_processor.has_received_command()) {
    return;
  }

  const bool host_overdue = (now_ms - command_processor.last_command_ms()) >= COMMAND_TIMEOUT_MS;

  if (host_overdue && !failsafe_engaged) {
    channels.engage_safe_state();
    failsafe_engaged = true;
    Serial.println("FAILSAFE outputs zeroed, no command received from host");

  } else if (!host_overdue && failsafe_engaged) {
    failsafe_engaged = false;
    Serial.println("FAILSAFE cleared, host link restored");
  }
}

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
  enforce_command_timeout(now_ms);

  delay(1);
}
