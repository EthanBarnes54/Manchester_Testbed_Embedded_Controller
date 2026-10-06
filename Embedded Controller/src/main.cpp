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
#include "driver/ledc.h"
#include "soc/gpio_reg.h"
#include "soc/gpio_sig_map.h"
#include "soc/gpio_struct.h"
#include "soc/ledc_struct.h"
#include "soc/soc.h"
#include "switch_timing.h"

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

// At the library's default 128 SPS a conversion takes 1/128 s, 7.8 ms nominal, and the
// ADS1115's oscillator is good to about 10%, so nothing is polled for before 7 ms.
// One not finished after the timeout is treated as lost: the ADC is absent or has
// dropped off the bus.
constexpr unsigned long ADC_FIRST_POLL_MS = 7;
constexpr unsigned long ADC_CONVERSION_TIMEOUT_MS = 40;

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

// SWITCH_PERIOD_US is the time between edges, half a cycle. Up to SWITCH_HARDWARE_MAX_US
// an LEDC channel generates the line with no CPU work per edge. Above it the timer
// interrupt toggles the pin: edges are then at least 1 ms apart, so the cost is
// negligible, but each edge can move by the interrupt latency (microseconds).
//
// The 1 us floor (a 500 kHz square wave) is the smallest whole period, and the LEDC is
// exact there: one counter bit, divider 1 on REF_TICK. It has NOT been scoped on
// hardware; what the downstream driver can follow is the real limit. 5 us is the
// requirement the rest of the stack was built to.
constexpr int SWITCH_PERIOD_MIN_US = 1;
constexpr int SWITCH_PERIOD_MAX_US = 2000000;
constexpr int SWITCH_HARDWARE_MAX_US = 1000;

constexpr uint32_t SWITCH_PIN_MASK = (1UL << SWITCH_LOGIC_PIN);

// The Arduino core maps LEDC channel c to speed group c / 8 and timer (c / 2) % 4, so the
// setpoints on channels 0-4 hold high-speed timers 0-2 and channel 6 has timer 3 to
// itself. Changing the switch period therefore cannot move the setpoint PWM frequency.
// Channel 5 would have shared cone_2's timer.
constexpr int SWITCH_LEDC_CHANNEL = 6;

constexpr int ledc_timer_of(int channel) {
  return (channel / 8) * 4 + (channel / 2) % 4;
}

constexpr bool setpoint_shares_timer_with(int channel, int index = 0) {
  return index < CONTROLLED_PULSE_CHANNELS &&
         (ledc_timer_of(LED_CONTROL_CHANNELS[index]) == ledc_timer_of(channel) || setpoint_shares_timer_with(channel, index + 1));
}

static_assert(SWITCH_LEDC_CHANNEL < 8, "The switch must use a high-speed LEDC channel");
static_assert(!setpoint_shares_timer_with(SWITCH_LEDC_CHANNEL), "The switch must not share an LEDC timer with a setpoint");

constexpr ledc_mode_t SWITCH_LEDC_MODE = LEDC_HIGH_SPEED_MODE;
constexpr ledc_timer_t SWITCH_LEDC_TIMER = static_cast<ledc_timer_t>((SWITCH_LEDC_CHANNEL / 2) % 4);
constexpr ledc_channel_t SWITCH_LEDC_HW_CHANNEL = static_cast<ledc_channel_t>(SWITCH_LEDC_CHANNEL);
constexpr uint32_t SWITCH_LEDC_SIGNAL = LEDC_HS_SIG_OUT0_IDX + SWITCH_LEDC_CHANNEL;

// REF_TICK is APB / 80 (APB_CTRL_PLL_TICK_NUM, default 79), a 1 MHz clock. On it every
// whole period up to 1024 us has an exact whole divider, where APB itself runs out at
// 205 us (switch_timing.h and tests/test_switch_timing.py).
//
// The CPU runs at 240 MHz (F_CPU from the esp32dev board definition) with APB at 80 MHz.
// APB must stay there: CONFIG_PM_ENABLE is unset in this core's sdkconfig and power
// management or frequency scaling must stay off, because changing APB would disturb
// REF_TICK and the 1 us tick of the switch timer interrupt.
constexpr uint32_t SWITCH_LEDC_CLOCK_HZ = REF_CLK_FREQ;

// Slack on top of two cycles before a new LEDC setting is declared stuck.
constexpr unsigned long SWITCH_LEDC_SETTLE_MARGIN_US = 100;

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

// The switch line on GPIO 16, always in one of three modes:
//   Held      - a fixed level on the plain GPIO output register. Off is held low.
//   Hardware  - an LEDC channel generates the square wave, with no CPU work per edge.
//   Interrupt - the timer interrupt toggles the pin, for periods above SWITCH_HARDWARE_MAX_US.
// Every change passes through Held low, so the line is low whenever nothing is driving
// it on purpose, including at boot and when the failsafe trips.
class SwitchLine {
 public:
  // GPIO 16 floats from reset until this runs, so setup() calls it before anything else.
  // The output register is cleared before the output is enabled, so the pin goes
  // straight from floating to low.
  static void hold_low_at_boot() {
    GPIO.out_w1tc = SWITCH_PIN_MASK;
    pinMode(SWITCH_LOGIC_PIN, OUTPUT);
  }

  // Needs the LEDC driver already up for the high-speed group, which the setpoint
  // ledcSetup() calls do.
  void begin() {
    instance_ = this;
    configure_ledc();
    configure_timer();
    hold(false);
  }

  // Stops any automatic switching and holds the line at a fixed level. This is also the
  // failsafe path, so it is ordered to be safe whatever was running: the level is set
  // and the pin taken back onto the output register before either generator is stopped,
  // so neither can reach the pin afterwards. A timer interrupt already pending when this
  // runs finds the line Held and leaves it alone.
  void hold(bool high) {
    portENTER_CRITICAL(&mux_);
    mode_ = Mode::Held;
    held_high_ = high;
    write_pin(high);
    route_pin(SIG_GPIO_OUT_IDX);
    portEXIT_CRITICAL(&mux_);

    if (timer_ != nullptr) {
      timerAlarmDisable(timer_);
    }

    if (ledc_ready_) {
      ledc_timer_pause(SWITCH_LEDC_MODE, SWITCH_LEDC_TIMER);
    }
  }

  // Switches with period_us between edges. On false the line is held low, or untouched
  // if nothing had been changed yet.
  bool start(unsigned long period_us) {
    if (period_us <= static_cast<unsigned long>(SWITCH_HARDWARE_MAX_US)) {
      return start_hardware(period_us);
    }

    return start_interrupt(period_us);
  }

  // switch_logic in PINS: the held level, or 1 while switching automatically. The CPU no
  // longer sees individual edges, so 1 means "not held low".
  int reported_level() const {
    portENTER_CRITICAL(const_cast<portMUX_TYPE*>(&mux_));
    const int level = (mode_ != Mode::Held || held_high_) ? 1 : 0;
    portEXIT_CRITICAL(const_cast<portMUX_TYPE*>(&mux_));
    return level;
  }

 private:
  enum class Mode { Held, Hardware, Interrupt };

  static SwitchLine* instance_;

  portMUX_TYPE mux_ = portMUX_INITIALIZER_UNLOCKED;
  Mode mode_ = Mode::Held;
  bool held_high_ = false;
  bool interrupt_high_ = false;
  bool ledc_ready_ = false;
  hw_timer_t* timer_ = nullptr;

  bool start_hardware(unsigned long period_us) {
    const switch_timing::LedcWaveform waveform = switch_timing::ledc_waveform_for(period_us, SWITCH_LEDC_CLOCK_HZ);

    // Every period this path accepts is exact (tests/test_switch_timing.py sweeps the
    // range), so a rejection here means the bounds were moved without the tests.
    if (!ledc_ready_ || !waveform.exact) {
      return false;
    }

    hold(false);

    // Everything is set up with the pin still on the output register. ledc_timer_set()
    // writes the whole divider straight into the register, and reading it back confirms
    // the hardware holds exactly what was asked for before any of it reaches the pin.
    ledc_timer_set(SWITCH_LEDC_MODE, SWITCH_LEDC_TIMER, waveform.divider_register(), waveform.counter_bits, LEDC_REF_TICK);
    ledc_set_duty_with_hpoint(SWITCH_LEDC_MODE, SWITCH_LEDC_HW_CHANNEL, waveform.high_counts, 0);
    ledc_update_duty(SWITCH_LEDC_MODE, SWITCH_LEDC_HW_CHANNEL);

    if (!timer_holds(waveform) || !run_one_cycle_off_pin(period_us)) {
      ledc_timer_pause(SWITCH_LEDC_MODE, SWITCH_LEDC_TIMER);
      LOG_ERROR("ERROR: Switch LEDC timer did not take its settings!");
      return false;
    }

    // Restart at the top of a cycle and hand the pin over before the count moves, so the
    // line begins with a full high half-cycle.
    portENTER_CRITICAL(&mux_);
    ledc_timer_pause(SWITCH_LEDC_MODE, SWITCH_LEDC_TIMER);
    ledc_timer_rst(SWITCH_LEDC_MODE, SWITCH_LEDC_TIMER);
    route_pin(SWITCH_LEDC_SIGNAL);
    ledc_timer_resume(SWITCH_LEDC_MODE, SWITCH_LEDC_TIMER);
    mode_ = Mode::Hardware;
    portEXIT_CRITICAL(&mux_);
    return true;
  }

  bool start_interrupt(unsigned long period_us) {
    if (timer_ == nullptr) {
      return false;
    }

    hold(false);

    // The count restarts from zero, so the first edge comes one full period after the
    // line went low however long the timer had been running.
    portENTER_CRITICAL(&mux_);
    interrupt_high_ = false;
    mode_ = Mode::Interrupt;
    timerWrite(timer_, 0);
    timerAlarmWrite(timer_, period_us, true);
    timerAlarmEnable(timer_);
    portEXIT_CRITICAL(&mux_);
    return true;
  }

  static bool timer_holds(const switch_timing::LedcWaveform& waveform) {
    const auto& conf = LEDC.timer_group[SWITCH_LEDC_MODE].timer[SWITCH_LEDC_TIMER].conf;

    // tick_sel 0 is REF_TICK, 1 is APB (ledc_struct.h).
    return conf.clock_divider == waveform.divider_register() && conf.duty_resolution == waveform.counter_bits &&
           conf.tick_sel == 0;
  }

  // Runs the timer for one whole cycle while the pin is still on the output register.
  // A channel only takes up a new duty and hpoint at the start of a cycle (ledc.h, on
  // ledc_update_duty), so this makes the first cycle the pin sees use the new settings,
  // and keeps the line low for at least that cycle on the way. Takes two periods: 10 us
  // at the floor, 2 ms at SWITCH_HARDWARE_MAX_US, and only when a period is set.
  static bool run_one_cycle_off_pin(unsigned long period_us) {
    const uint32_t overflow = BIT(SWITCH_LEDC_TIMER);

    ledc_timer_rst(SWITCH_LEDC_MODE, SWITCH_LEDC_TIMER);
    LEDC.int_clr.val = overflow;
    ledc_timer_resume(SWITCH_LEDC_MODE, SWITCH_LEDC_TIMER);

    const unsigned long started_us = micros();

    while ((LEDC.int_raw.val & overflow) == 0) {
      if (micros() - started_us > 4UL * period_us + SWITCH_LEDC_SETTLE_MARGIN_US) {
        return false;
      }
    }

    return true;
  }

  void configure_ledc() {
    ledc_ready_ = ledc_timer_pause(SWITCH_LEDC_MODE, SWITCH_LEDC_TIMER) == ESP_OK &&
                  ledc_bind_channel_timer(SWITCH_LEDC_MODE, SWITCH_LEDC_HW_CHANNEL, SWITCH_LEDC_TIMER) == ESP_OK;

    if (ledc_ready_) {
      LOG_INFO("Switch logic LEDC channel initialised...");
    } else {
      LOG_ERROR("ERROR: Hardware switching unavailable! (LEDC not initialised)");
    }
  }

  void configure_timer() {
    timer_ = timerBegin(0, 80, true);

    if (timer_ != nullptr) {
      timerAttachInterrupt(timer_, &SwitchLine::on_timer, true);
      timerAlarmDisable(timer_);
      LOG_INFO("Switch logic timer initialised...");
    } else {
      LOG_ERROR("ERROR: Switch unavailable! (Timer allocation failed)");
    }
  }

  static void IRAM_ATTR on_timer() {
    if (instance_ != nullptr) {
      instance_->toggle();
    }
  }

  void IRAM_ATTR toggle() {
    portENTER_CRITICAL_ISR(&mux_);

    if (mode_ == Mode::Interrupt) {
      interrupt_high_ = !interrupt_high_;
      write_pin(interrupt_high_);
    }

    portEXIT_CRITICAL_ISR(&mux_);
  }

  static void write_pin(bool high) {
    if (high) {
      GPIO.out_w1ts = SWITCH_PIN_MASK;
    } else {
      GPIO.out_w1tc = SWITCH_PIN_MASK;
    }
  }

  // Gives GPIO 16 to the plain output register or to the LEDC channel in one register
  // write, so the pin never passes through a half-set routing. Output enable always comes
  // from GPIO_ENABLE (oen_sel), which pinMode() set at boot.
  static void route_pin(uint32_t signal) {
    GPIO.func_out_sel_cfg[SWITCH_LOGIC_PIN].val = signal | GPIO_FUNC0_OEN_SEL;
  }
};

SwitchLine* SwitchLine::instance_ = nullptr;

class ChannelController {
 public:
  ChannelController() = default;

  void begin() {
    init_pwm_channels();
    switch_line_.begin();
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
    Serial.println(String("ACK PIN ") + channel_number + " " + switch_line_.reported_level());
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

  // switch_logic is 0 or 1: the held level, or 1 while the line switches automatically.
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
    return switch_line_.start(period_us);
  }

  void stop_switching(int switch_level) {
    switch_line_.hold(switch_level != 0);
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

    memcpy(destination, channel_values_, sizeof(channel_values_));
    destination[5] = switch_line_.reported_level();
  }

 private:
  // Setpoints only. The switch line keeps its own state.
  int channel_values_[CHANNEL_COUNT] = {0};
  SwitchLine switch_line_;

  void init_pwm_channels() {
    for (int i = 0; i < CONTROLLED_PULSE_CHANNELS; ++i) {
      pinMode(CHANNEL_PINS[i], OUTPUT);
      ledcSetup(LED_CONTROL_CHANNELS[i], MODULATION_FREQUENCY, MODULATION_RESOLUTION);
      ledcAttachPin(CHANNEL_PINS[i], LED_CONTROL_CHANNELS[i]);
      ledcWrite(LED_CONTROL_CHANNELS[i], 0);
      channel_values_[i] = 0;
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

class MeasurementService {
 public:
  explicit MeasurementService(Adafruit_ADS1115& adc) : adc_(adc) {}

  bool begin() {
    adc_.setGain(ADC_GAIN);
    return adc_.begin();
  }

  // Single formatter for every MEASURED line, so the streamed and on-demand readings
  // always carry the same precision.
  void report_voltage(float volts) {
    Serial.println(String("MEASURED ") + String(volts, MEASURED_DECIMAL_PLACES) + " V");
  }

  // READ is answered by the next conversion to complete. One already in flight is never
  // restarted, so a READ cannot corrupt it; with the ADC idle a conversion starts at once.
  void request_reading() {
    reading_requested_ = true;
  }

  // Starts a conversion when one is due and collects it once the ADC reports it done, so
  // the loop never waits on one. The library's readADC_SingleEnded() polls until the ADC
  // answers, and with the ADC missing or off the bus it never returns, which used to stop
  // the loop and the failsafe with it.
  void update(unsigned long now_ms) {
    if (!converting_) {
      if (reading_requested_ || now_ms - conversion_started_ms_ >= MEASUREMENT_INTERVAL_MS) {
        adc_.startADCReading(MUX_BY_CHANNEL[0], /*continuous=*/false);
        converting_ = true;
        conversion_started_ms_ = now_ms;
      }

      return;
    }

    const unsigned long elapsed_ms = now_ms - conversion_started_ms_;

    if (elapsed_ms < ADC_FIRST_POLL_MS) {
      return;
    }

    if (adc_.conversionComplete()) {
      converting_ = false;
      reading_requested_ = false;
      fault_reported_ = false;
      report_voltage(adc_.computeVolts(adc_.getLastConversionResults()));
      return;
    }

    if (elapsed_ms >= ADC_CONVERSION_TIMEOUT_MS) {
      converting_ = false;

      // Once per outage rather than at 20 Hz, but always in answer to a READ.
      if (!fault_reported_ || reading_requested_) {
        Serial.println("ERROR: ADC conversion timed out!");
      }

      reading_requested_ = false;
      fault_reported_ = true;
    }
  }

 private:
  Adafruit_ADS1115& adc_;
  unsigned long conversion_started_ms_ = 0;
  bool converting_ = false;
  bool reading_requested_ = false;
  bool fault_reported_ = false;
};

// Blinks are scheduled rather than slept through - update() plays them out from loop(),
// so an LED pattern can never hold up serial commands, sampling or the failsafe.
class LedIndicator {
 public:
  explicit LedIndicator(int pin) : pin_(pin) {}

  void begin() {
    pinMode(pin_, OUTPUT);
    digitalWrite(pin_, HIGH);
  }

  void heartbeat(unsigned long now_ms) {
    if (now_ms - last_heartbeat_ms_ > HEARTBEAT_INTERVAL_MS) {
      last_heartbeat_ms_ = now_ms;

      // A command or error pattern already in flight takes precedence over the beat.
      if (!busy()) {
        start_pattern(1, HEARTBEAT_FLASH_MS, 0, now_ms);
      }
    }
  }

  void blink_once(int duration_ms = 100) {
    start_pattern(1, duration_ms, 0, millis());
  }

  void blink_error(int flash_count = 3, int duration_ms = 100) {
    start_pattern(flash_count, duration_ms, duration_ms, millis());
  }

  void update(unsigned long now_ms) {
    // Loops so a zero-length phase is skipped in the same call rather than costing a pass.
    while (phases_remaining_ > 0) {
      if (now_ms - phase_started_ms_ < current_phase_ms()) {
        return;
      }

      --phases_remaining_;
      phase_started_ms_ = now_ms;
      digitalWrite(pin_, phase_is_on() ? HIGH : LOW);
    }
  }

  // Ends the steady boot light, but leaves any pattern already scheduled to finish.
  void set_low() {
    if (!busy()) {
      digitalWrite(pin_, LOW);
    }
  }

  bool busy() const {
    return phases_remaining_ > 0;
  }

 private:
  static constexpr unsigned long HEARTBEAT_FLASH_MS = 50;

  int pin_;
  unsigned long last_heartbeat_ms_ = 0;

  // A pattern is 2 * flashes phases counted down to zero: even counts are lit, odd are dark.
  unsigned int phases_remaining_ = 0;
  unsigned long phase_started_ms_ = 0;
  unsigned long on_ms_ = 0;
  unsigned long off_ms_ = 0;

  bool phase_is_on() const {
    return phases_remaining_ > 0 && (phases_remaining_ % 2) == 0;
  }

  unsigned long current_phase_ms() const {
    return phase_is_on() ? on_ms_ : off_ms_;
  }

  void start_pattern(int flash_count, unsigned long on_ms, unsigned long off_ms, unsigned long now_ms) {
    if (flash_count <= 0) {
      return;
    }

    phases_remaining_ = 2U * static_cast<unsigned int>(flash_count);
    on_ms_ = on_ms;
    off_ms_ = off_ms;
    phase_started_ms_ = now_ms;
    digitalWrite(pin_, HIGH);
  }
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
      measurement_.request_reading();
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
  SwitchLine::hold_low_at_boot();
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
  measurement_service.update(now_ms);
  led_indicator.heartbeat(now_ms);
  led_indicator.update(now_ms);
  ota_wifi_service.loop();
  enforce_command_timeout(now_ms);

  delay(1);
}
