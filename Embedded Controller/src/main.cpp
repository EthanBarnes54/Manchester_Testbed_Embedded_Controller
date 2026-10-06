// ---------------- Pin control (5 PWM-like + 1 digital) ---------------- //
// Pin Channels:
//  1) squeeze_plate  (PWM 0..1023)
//  2) ion_source     (PWM 0..1023)
//  3) wein_filter    (PWM 0..1023)
//  4) cone_1         (PWM 0..1023)
//  5) cone_2         (PWM 0..1023)
//  6) switch_logic   (digital 0/1)
//
// GPIO 23 (ARMED) enables the external AND gate between switch_logic and its load.

#include <Arduino.h>
#include "log.h"
#include <Wire.h>
#include <cstring>
#include <WiFi.h>
#include <ESPmDNS.h>
#include <WiFiUdp.h>
#include <ArduinoOTA.h>
#include <Adafruit_ADS1X15.h>
#include <Preferences.h>
#include "esp_ota_ops.h"
#include "esp_system.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "driver/ledc.h"
#include "driver/pcnt.h"
#include "soc/apb_ctrl_reg.h"
#include "soc/gpio_reg.h"
#include "soc/gpio_sig_map.h"
#include "soc/gpio_struct.h"
#include "soc/ledc_reg.h"
#include "soc/ledc_struct.h"
#include "soc/soc.h"
#include "health_stats.h"
#include "line_protocol.h"
#include "output_checks.h"
#include "safety_state.h"
#include "switch_timing.h"

namespace{

constexpr int LED_PIN = 2;

constexpr int SQUEEZE_PLATE_PIN = 25;
constexpr int ION_SOURCE_PIN = 26;
constexpr int WEIN_FILTER_PIN = 27;
constexpr int CONE_1_PIN = 32;
constexpr int CONE_2_PIN = 33;
constexpr int SWITCH_LOGIC_PIN = 16;

// Second input of the AND gate between GPIO 16 and the load, with a pull-down on the board.
// GPIO 23 is not a strapping pin, and neither the bootloader nor the PSRAM probe touches
// it, unlike GPIO 16 (the PSRAM chip-select on this chip).
constexpr int SWITCH_ARMED_PIN = 23;

// Toggled every loop pass for an external watchdog (for example a TPS3823) whose output
// also enables the gate. If the loop stops, the gate closes within the watchdog's
// timeout, without relying on this processor. Harmless with nothing fitted.
constexpr int HEARTBEAT_PIN = 18;

// The gate output looped back for output verification, when fitted (see below). An
// input-only pin, untouched at boot.
constexpr int GATE_LOOPBACK_PIN = 34;

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

// Output verification (README, "Output verification"). Each check is compiled in but
// stays off until its build flag says the hardware is fitted, so a bench without it
// cannot raise false faults. The deploy environment turns both on.
#ifndef TESTBED_GATE_LOOPBACK
#define TESTBED_GATE_LOOPBACK 0
#endif

#ifndef TESTBED_SETPOINT_READBACK
#define TESTBED_SETPOINT_READBACK 0
#endif

constexpr bool GATE_LOOPBACK_FITTED = TESTBED_GATE_LOOPBACK != 0;
constexpr bool SETPOINT_READBACK_FITTED = TESTBED_SETPOINT_READBACK != 0;

// Gate loopback: the pulse counter counts rising edges on GATE_LOOPBACK_PIN. Its filter
// drops anything shorter than 125 ns; a 1 us pulse is 80 APB cycles.
constexpr pcnt_unit_t LOOPBACK_PCNT_UNIT = PCNT_UNIT_0;
constexpr uint16_t LOOPBACK_FILTER_APB_CYCLES = 10;
constexpr uint8_t GATE_MISMATCH_PASSES = 3;

// Setpoint readback: ADS1115 AIN1-AIN3 read back setpoints 1-3 (squeeze_plate, ion_source,
// wein_filter) through whatever network the board uses, scaled by READBACK_VOLTS_PER_VOLT.
// cone_1 and cone_2 need a second ADS1115 (address 0x49) to be covered. A reading only
// counts once the setpoint has had READBACK_SETTLE_MS to settle after a change.
constexpr uint8_t READBACK_INPUTS = 3;
constexpr uint8_t READBACK_SETPOINTS[READBACK_INPUTS] = {1, 2, 3};
constexpr float READBACK_VOLTS_PER_VOLT = 1.0f;
constexpr float READBACK_TOLERANCE_V = 0.15f;
constexpr unsigned long READBACK_SETTLE_MS = 500;
constexpr uint8_t READBACK_MISMATCH_READINGS = 3;

// A readback conversion only starts this soon after a diode conversion started, so even
// one that times out has finished before the next diode conversion is due.
constexpr unsigned long READBACK_START_WINDOW_MS = MEASUREMENT_INTERVAL_MS - ADC_CONVERSION_TIMEOUT_MS;

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

// Flash namespace for the fault log that survives resets (Preferences, at most 15 characters).
constexpr const char* FAULT_LOG_NAMESPACE = "testbed_faults";

// Continuous self-test budgets. A normal pass is about a millisecond plus the delay(1);
// the longest legitimate one is a switch period change priming the LEDC (two periods,
// 2 ms at 1 ms between edges). Twenty times that is a pass that has gone wrong.
constexpr uint32_t LOOP_BUDGET_US = 20000;
constexpr unsigned long HEALTH_CHECK_INTERVAL_MS = 1000;
constexpr uint32_t HEAP_FLOOR_BYTES = 32768;
constexpr uint32_t STACK_FLOOR_BYTES = 1024;

// SWITCH_PERIOD_US is the time between edges, half a cycle. Up to SWITCH_HARDWARE_MAX_US
// an LEDC channel generates the line with no CPU work per edge. Above it the timer
// interrupt toggles the pin: edges are then at least 1 ms apart, so the cost is
// negligible, but each edge can move by the interrupt latency (microseconds).
//
// 1 us between edges (a 500 kHz square wave) is the design floor, and the LEDC is exact
// there: one counter bit, divider 1 on REF_TICK. It has not been scoped on hardware yet.
// What the gate's load can follow is the real limit, and that load is still to be
// decided (README, "Switch output stage").
constexpr int SWITCH_PERIOD_MIN_US = 1;
constexpr int SWITCH_PERIOD_MAX_US = 2000000;
constexpr int SWITCH_HARDWARE_MAX_US = 1000;

constexpr uint32_t SWITCH_PIN_MASK = (1UL << SWITCH_LOGIC_PIN);
constexpr uint32_t SWITCH_ARMED_PIN_MASK = (1UL << SWITCH_ARMED_PIN);

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

// A production build (the esp32_deploy environment) never starts the radio: no Wi-Fi, no
// OTA, nothing listening. Updates then go over the cable only.
#ifndef TESTBED_PRODUCTION
#define TESTBED_PRODUCTION 0
#endif

constexpr bool WIRELESS_ENABLED = TESTBED_PRODUCTION == 0;

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

// The serial protocol's version. Bump it whenever a line's format or meaning changes: the
// backend will not arm a board whose VERSION reports a different one.
constexpr int PROTOCOL_VERSION = 2;

// Stamped by tools/firmware_version.py from git and the PlatformIO environment.
#ifndef TESTBED_FIRMWARE_VERSION
#define TESTBED_FIRMWARE_VERSION "unknown"
#endif

#ifndef TESTBED_BUILD_ENV
#define TESTBED_BUILD_ENV "unknown"
#endif

}

// Every line to the host goes through here and leaves with a CRC (line_protocol.h), so the
// backend can tell a corrupted line from a real one. Only the debug LOG_ macros and the OTA
// progress counter bypass it.
void send_line(const String& line) {
  char suffix[line_protocol::CHECKSUM_LENGTH + 1];
  line_protocol::format_suffix(line_protocol::crc16(line.c_str(), line.length()), suffix);
  Serial.print(line);
  Serial.println(suffix);
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
  // GPIO 16 and ARMED float from reset until this runs, so setup() calls it before
  // anything else. The output register is cleared before the outputs are enabled, so both
  // pins go straight from floating to low.
  static void hold_low_at_boot() {
    GPIO.out_w1tc = SWITCH_PIN_MASK | SWITCH_ARMED_PIN_MASK;
    pinMode(SWITCH_LOGIC_PIN, OUTPUT);
    pinMode(SWITCH_ARMED_PIN, OUTPUT);
  }

  // ARMED opens the external AND gate that passes GPIO 16 to the load. It is low from
  // reset, held there by the board's pull-down until this firmware drives it, so nothing
  // the chip does to GPIO 16 while booting can reach the load. The failsafe drops it as a
  // second path to low, independent of GPIO 16. It never changes the switch mode, and
  // nothing in the switch state machine touches it.
  static void set_armed(bool armed) {
    if (armed) {
      GPIO.out_w1ts = SWITCH_ARMED_PIN_MASK;
    } else {
      GPIO.out_w1tc = SWITCH_ARMED_PIN_MASK;
    }
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
    period_us_ = 0;
    ++generation_;
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

  // What the line is doing, read consistently: switching or held, the held level, the
  // period, and a generation that changes on every transition so a measurement spanning
  // one can be thrown away.
  struct Snapshot {
    bool switching;
    bool held_high;
    uint32_t period_us;
    uint32_t generation;
  };

  Snapshot snapshot() const {
    portENTER_CRITICAL(const_cast<portMUX_TYPE*>(&mux_));
    const Snapshot snapshot = {mode_ != Mode::Held, held_high_, period_us_, generation_};
    portEXIT_CRITICAL(const_cast<portMUX_TYPE*>(&mux_));
    return snapshot;
  }

  // Both generators came up in begin(). Checked by the power-on self-test.
  bool generators_ready() const {
    return ledc_ready_ && timer_ != nullptr;
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
  uint32_t period_us_ = 0;
  uint32_t generation_ = 0;
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

    portENTER_CRITICAL(&mux_);
    hand_pin_to_ledc();
    mode_ = Mode::Hardware;
    period_us_ = period_us;
    ++generation_;
    portEXIT_CRITICAL(&mux_);
    return true;
  }

  // Restarts the timer at the top of a cycle and gives it the pin. The pin is routed while
  // the count is held at zero and released by the next store, so the first high
  // half-cycle is stretched by tens of nanoseconds at most. Going through the driver here
  // put its argument checks and spinlock between the two, which at 1 us between edges is a
  // large share of the first pulse. Kept out of line in IRAM, so a flash cache miss
  // cannot land between the two stores either. Pause and reset sit at the same bits in
  // every high-speed timer's conf register.
  static void NOINLINE_ATTR IRAM_ATTR hand_pin_to_ledc() {
    auto& conf = LEDC.timer_group[SWITCH_LEDC_MODE].timer[SWITCH_LEDC_TIMER].conf;
    const uint32_t held = conf.val | LEDC_HSTIMER0_PAUSE;

    conf.val = held;
    conf.val = held | LEDC_HSTIMER0_RST;
    conf.val = held;
    route_pin(SWITCH_LEDC_SIGNAL);
    conf.val = held & ~LEDC_HSTIMER0_PAUSE;
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
    period_us_ = period_us;
    ++generation_;
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
  static void IRAM_ATTR route_pin(uint32_t signal) {
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
      send_line("ERROR: PIN index out of range!");
      return;
    }

    const uint8_t channel_index = channel_number - 1;

    if (channel_index < CONTROLLED_PULSE_CHANNELS) {
      if (value < 0) value = 0;
      if (value > MAX_MODULATION_VALUE) value = MAX_MODULATION_VALUE;

      ledcWrite(LED_CONTROL_CHANNELS[channel_index], value);
      note_setpoint(channel_index, value);
      send_line(String("ACK PIN ") + channel_number + " " + value);
      return;
    }

    stop_switching(value ? 1 : 0);
    send_line(String("ACK PIN ") + channel_number + " " + switch_line_.reported_level());
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

  // True when these TARGETS would drive any channel above zero. A line that does not
  // parse is left for apply_target_voltages() to reject.
  static bool targets_would_energise(const String& args) {
    float targets[CONTROLLED_PULSE_CHANNELS];

    if (!parse_voltage_targets(args, targets, CONTROLLED_PULSE_CHANNELS)) {
      return false;
    }

    for (int i = 0; i < CONTROLLED_PULSE_CHANNELS; ++i) {
      if (convert_voltage_to_pwm(targets[i]) > 0) {
        return true;
      }
    }

    return false;
  }

  // switch_logic is 0 or 1: the held level, or 1 while the line switches automatically.
  void report_channels() const {
    int snapshot[CHANNEL_COUNT];
    snapshot_values(snapshot, CHANNEL_COUNT);

    send_line(String("PINS ") +
              "squeeze_plate=" + snapshot[0] + " " +
              "ion_source="    + snapshot[1] + " " +
              "wein_filter="   + snapshot[2] + " " +
              "cone_1="        + snapshot[3] + " " +
              "cone_2="        + snapshot[4] + " " +
              "switch_logic="  + snapshot[5]);
  }

  bool automate_switching(unsigned long period_us) {
    return switch_line_.start(period_us);
  }

  void stop_switching(int switch_level) {
    switch_line_.hold(switch_level != 0);
  }

  bool switch_generators_ready() const {
    return switch_line_.generators_ready();
  }

  SwitchLine::Snapshot switch_snapshot() const {
    return switch_line_.snapshot();
  }

  // Setpoint channel_number (1-5) as last written, and when it last changed.
  int setpoint_value(uint8_t channel_number) const {
    return channel_values_[channel_number - 1];
  }

  unsigned long setpoint_changed_ms(uint8_t channel_number) const {
    return setpoint_changed_ms_[channel_number - 1];
  }

  // The external gate on the switch line. Opened only by SystemSupervisor on ARM, closed
  // by engage_safe_state().
  void set_switch_armed(bool armed) {
    SwitchLine::set_armed(armed);
  }

  // Drops every output back to zero and stops the switch line. Used when the host
  // stops talking to us, so a latched target cannot outlive the controlling process.
  // The gate closes first, which takes the switch off the load in one register write.
  void engage_safe_state() {
    set_switch_armed(false);

    for (int i = 0; i < CONTROLLED_PULSE_CHANNELS; ++i) {
      ledcWrite(LED_CONTROL_CHANNELS[i], 0);
      note_setpoint(i, 0);
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
  unsigned long setpoint_changed_ms_[CONTROLLED_PULSE_CHANNELS] = {0};
  SwitchLine switch_line_;

  void note_setpoint(int channel_index, int value) {
    if (channel_values_[channel_index] != value) {
      setpoint_changed_ms_[channel_index] = millis();
    }

    channel_values_[channel_index] = value;
  }

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
    lost_ = !adc_.begin();
    return !lost_;
  }

  // The ADC is absent, or its last conversion timed out. Cleared by the next conversion
  // that completes, so an ADC that comes back is noticed without a reset.
  bool lost() const {
    return lost_;
  }

  uint32_t conversions() const {
    return conversions_;
  }

  uint32_t timeouts() const {
    return timeouts_;
  }

  // Setpoint readback (when fitted): the latest reading on ADS1115 input 1-3 and when it
  // was taken, or 0 if there has not been one yet.
  float readback_volts(uint8_t input) const {
    return readback_volts_[input];
  }

  unsigned long readback_ms(uint8_t input) const {
    return readback_ms_[input];
  }

  // Single formatter for every MEASURED line, so the streamed and on-demand readings
  // always carry the same precision. seq counts readings from 1 at boot, so the host can
  // see one go missing or the board restart; t_ms is the board's clock when it was taken.
  void report_voltage(float volts) {
    ++sequence_;
    send_line(String("MEASURED ") + String(volts, MEASURED_DECIMAL_PLACES) + " V seq=" + sequence_ + " t_ms=" + millis());
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
  //
  // The diode on AIN0 keeps its 50 ms schedule. With setpoint readback fitted, one readback
  // conversion (AIN1-AIN3 in turn) fits in the gap after each diode conversion, so each
  // setpoint is read about every 150 ms and the diode stream is never delayed.
  void update(unsigned long now_ms) {
    if (!converting_) {
      if (reading_requested_ || now_ms - diode_started_ms_ >= MEASUREMENT_INTERVAL_MS) {
        start_conversion(0, now_ms);
        diode_started_ms_ = now_ms;
        readback_due_ = SETPOINT_READBACK_FITTED;
      } else if (readback_due_ && now_ms - diode_started_ms_ <= READBACK_START_WINDOW_MS) {
        start_conversion(next_readback_input_, now_ms);
        readback_due_ = false;
      }

      return;
    }

    const unsigned long elapsed_ms = now_ms - conversion_started_ms_;

    if (elapsed_ms < ADC_FIRST_POLL_MS) {
      return;
    }

    if (adc_.conversionComplete()) {
      converting_ = false;
      fault_reported_ = false;
      lost_ = false;
      ++conversions_;
      const float volts = adc_.computeVolts(adc_.getLastConversionResults());

      if (input_ == 0) {
        reading_requested_ = false;
        report_voltage(volts);
      } else {
        readback_volts_[input_] = volts;
        readback_ms_[input_] = now_ms;
        next_readback_input_ = static_cast<uint8_t>(next_readback_input_ % READBACK_INPUTS + 1);
      }

      return;
    }

    if (elapsed_ms >= ADC_CONVERSION_TIMEOUT_MS) {
      converting_ = false;

      // Once per outage rather than at 20 Hz, but always in answer to a READ.
      if (!fault_reported_ || reading_requested_) {
        send_line("ERROR: ADC conversion timed out!");
      }

      reading_requested_ = false;
      fault_reported_ = true;
      lost_ = true;
      ++timeouts_;
    }
  }

 private:
  Adafruit_ADS1115& adc_;
  unsigned long conversion_started_ms_ = 0;
  unsigned long diode_started_ms_ = 0;
  uint8_t input_ = 0;
  bool readback_due_ = false;
  uint8_t next_readback_input_ = 1;
  float readback_volts_[READBACK_INPUTS + 1] = {0};
  unsigned long readback_ms_[READBACK_INPUTS + 1] = {0};
  bool converting_ = false;
  bool reading_requested_ = false;
  bool fault_reported_ = false;
  bool lost_ = false;
  uint32_t conversions_ = 0;
  uint32_t timeouts_ = 0;
  uint32_t sequence_ = 0;

  void start_conversion(uint8_t input, unsigned long now_ms) {
    adc_.startADCReading(MUX_BY_CHANNEL[input], /*continuous=*/false);
    converting_ = true;
    conversion_started_ms_ = now_ms;
    input_ = input;
  }
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
      send_line("WARNING: WiFi credentials not set, running on serial only...");
      return;
    }

    ota_allowed_ = strlen(OTA_PASSWORD) > 0;

    if (ota_allowed_) {
      ArduinoOTA.setPassword(OTA_PASSWORD);
    } else {
      send_line("WARNING: OTA password not set, over the air updates disabled...");
    }

    ArduinoOTA.setHostname(hostname);

    // An upload holds the loop for its whole length without feeding the loop watchdog,
    // so the watchdog stands down for it. Uploads are only serviced while disarmed (see
    // loop()), so nothing is live while the loop is held.
    ArduinoOTA.onStart([]() {
      disableLoopWDT();
      send_line("OTA connection starting...");
    });

    ArduinoOTA.onEnd([]() {
      send_line("OTA connection established...");
      enableLoopWDT();
    });

    ArduinoOTA.onProgress([](unsigned int connection_progress, unsigned int connection_capacity) {
      if (connection_capacity) {
        Serial.printf("OTA Progress: %u%%\r", (connection_progress * 100) / connection_capacity);
      }
    });

    ArduinoOTA.onError([](ota_error_t error) {
      enableLoopWDT();

      if (error == OTA_AUTH_ERROR) send_line("ERROR: OTA authentication failed!");
      else if (error == OTA_BEGIN_ERROR) send_line("ERROR: OTA begin failed!");
      else if (error == OTA_CONNECT_ERROR) send_line("ERROR: OTA connect failed!");
      else if (error == OTA_RECEIVE_ERROR) send_line("ERROR: OTA receive failed!");
      else if (error == OTA_END_ERROR) send_line("ERROR: OTA end failed!");
      else send_line(String("ERROR: OTA failed with code ") + static_cast<int>(error) + "!");
    });

    WiFi.mode(WIFI_STA);
    WiFi.setAutoReconnect(true);
    start_association();
  }

  // updates_allowed is false while the board is armed: an OTA upload blocks the loop, and
  // with it the failsafe, for as long as it runs, so one is never accepted with outputs
  // live. The client is simply not answered and times out.
  void loop(bool updates_allowed) {
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

        if (ota_started_ && updates_allowed) {
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
    send_line("WiFi associating...");
  }

  void enter_online_state() {
    link_state_ = LinkState::Online;
    state_entered_ms_ = millis();

    if (ota_allowed_ && !ota_started_) {
      ArduinoOTA.begin();
      ota_started_ = true;
      send_line("Ready for OTA updates...");
    }

    send_line(String("WiFi connected, IP address: ") + WiFi.localIP().toString());
  }

  // Drops OTA on the way down so the next association rebinds against the new address.
  void enter_waiting_state(const char* reason) {
    if (ota_started_) {
      ArduinoOTA.end();
      ota_started_ = false;
    }

    link_state_ = LinkState::Waiting;
    state_entered_ms_ = millis();
    send_line(reason);
  }
};

// Checks the outputs against what was commanded, through hardware fitted for it (README,
// "Output verification"): the gate output looped back to GATE_LOOPBACK_PIN and counted by
// the pulse counter, and three setpoints read back on the ADS1115's spare inputs. Each
// part does nothing unless its build flag says the hardware is there.
class OutputVerifier {
 public:
  OutputVerifier(ChannelController& channels, MeasurementService& measurement)
      : channels_(channels),
        measurement_(measurement),
        gate_level_check_(GATE_MISMATCH_PASSES),
        readback_checks_{output_checks::Debounce(READBACK_MISMATCH_READINGS), output_checks::Debounce(READBACK_MISMATCH_READINGS),
                         output_checks::Debounce(READBACK_MISMATCH_READINGS)} {}

  void begin() {
    if (!GATE_LOOPBACK_FITTED) {
      return;
    }

    pinMode(GATE_LOOPBACK_PIN, INPUT);

    pcnt_config_t config = {};
    config.pulse_gpio_num = GATE_LOOPBACK_PIN;
    config.ctrl_gpio_num = PCNT_PIN_NOT_USED;
    config.lctrl_mode = PCNT_MODE_KEEP;
    config.hctrl_mode = PCNT_MODE_KEEP;
    config.pos_mode = PCNT_COUNT_INC;
    config.neg_mode = PCNT_COUNT_DIS;
    config.counter_h_lim = 32767;
    config.counter_l_lim = 0;
    config.unit = LOOPBACK_PCNT_UNIT;
    config.channel = PCNT_CHANNEL_0;

    counter_ready_ = pcnt_unit_config(&config) == ESP_OK && pcnt_set_filter_value(LOOPBACK_PCNT_UNIT, LOOPBACK_FILTER_APB_CYCLES) == ESP_OK &&
                     pcnt_filter_enable(LOOPBACK_PCNT_UNIT) == ESP_OK && pcnt_counter_clear(LOOPBACK_PCNT_UNIT) == ESP_OK;
    window_started_us_ = micros();
  }

  // Once per loop pass.
  void check(bool armed) {
    if (GATE_LOOPBACK_FITTED) {
      check_gate(armed);
    }

    if (SETPOINT_READBACK_FITTED) {
      check_readback();
    }
  }

  // The gate output disagrees with ARMED and the switch, or the loopback counter would not start.
  bool gate_fault() const {
    return GATE_LOOPBACK_FITTED && (!counter_ready_ || gate_level_check_.tripped() || stray_edges_);
  }

  bool frequency_fault() const {
    return GATE_LOOPBACK_FITTED && frequency_wrong_;
  }

  bool readback_fault() const {
    if (!SETPOINT_READBACK_FITTED) {
      return false;
    }

    for (const output_checks::Debounce& check : readback_checks_) {
      if (check.tripped()) {
        return true;
      }
    }

    return false;
  }

  const char* gate_verdict() const {
    return !GATE_LOOPBACK_FITTED ? "off" : (gate_fault() || frequency_fault() ? "FAIL" : "ok");
  }

  const char* readback_verdict() const {
    return !SETPOINT_READBACK_FITTED ? "off" : (readback_fault() ? "FAIL" : "ok");
  }

  uint32_t last_window_edges() const {
    return last_window_edges_;
  }

 private:
  ChannelController& channels_;
  MeasurementService& measurement_;
  output_checks::Debounce gate_level_check_;
  output_checks::Debounce readback_checks_[READBACK_INPUTS];
  unsigned long readback_checked_ms_[READBACK_INPUTS] = {0};
  bool counter_ready_ = false;
  bool frequency_wrong_ = false;
  bool stray_edges_ = false;
  unsigned long window_started_us_ = 0;
  uint32_t window_generation_ = 0;
  bool window_armed_ = false;
  uint32_t last_window_edges_ = 0;

  // Level every pass (debounced); edges over a window long enough to see about four
  // cycles. A window that spans a switch or arming transition is thrown away.
  void check_gate(bool armed) {
    const SwitchLine::Snapshot line = channels_.switch_snapshot();
    const output_checks::GateExpectation expected = output_checks::expected_gate(armed, line.switching, line.held_high);
    const bool high = digitalRead(GATE_LOOPBACK_PIN) == HIGH;

    if (expected != output_checks::GateExpectation::Switching) {
      gate_level_check_.update(high != (expected == output_checks::GateExpectation::High));
    } else {
      gate_level_check_.update(false);
    }

    const unsigned long now_us = micros();
    const unsigned long elapsed_us = now_us - window_started_us_;

    if (!counter_ready_ || elapsed_us < output_checks::frequency_window_us(line.period_us)) {
      return;
    }

    int16_t counted = 0;
    pcnt_get_counter_value(LOOPBACK_PCNT_UNIT, &counted);
    pcnt_counter_clear(LOOPBACK_PCNT_UNIT);
    last_window_edges_ = counted < 0 ? 0 : static_cast<uint32_t>(counted);

    const bool window_valid = line.generation == window_generation_ && armed == window_armed_;
    window_started_us_ = now_us;
    window_generation_ = line.generation;
    window_armed_ = armed;

    if (!window_valid) {
      return;
    }

    if (expected == output_checks::GateExpectation::Switching) {
      stray_edges_ = false;
      frequency_wrong_ = !output_checks::edge_count_plausible(
          last_window_edges_, output_checks::expected_rising_edges(elapsed_us, line.period_us));
    } else {
      // A closed gate or a held level passes no edges at all.
      stray_edges_ = last_window_edges_ > 0;
      frequency_wrong_ = false;
    }
  }

  // Each fresh reading taken at least READBACK_SETTLE_MS after its setpoint last changed is
  // compared with what that setpoint was commanded to. The difference is taken signed, so
  // a reading from before the change is skipped rather than wrapping round.
  void check_readback() {
    for (uint8_t index = 0; index < READBACK_INPUTS; ++index) {
      const uint8_t input = index + 1;
      const uint8_t setpoint = READBACK_SETPOINTS[index];
      const unsigned long read_ms = measurement_.readback_ms(input);

      if (read_ms == 0 || read_ms == readback_checked_ms_[index]) {
        continue;
      }

      readback_checked_ms_[index] = read_ms;

      if (static_cast<long>(read_ms - channels_.setpoint_changed_ms(setpoint)) < static_cast<long>(READBACK_SETTLE_MS)) {
        continue;
      }

      const float commanded_v = static_cast<float>(channels_.setpoint_value(setpoint)) / MAX_MODULATION_VALUE * 3.3f;
      readback_checks_[index].update(!output_checks::readback_matches(
          commanded_v * READBACK_VOLTS_PER_VOLT, measurement_.readback_volts(input), READBACK_TOLERANCE_V));
    }
  }
};

// Owns the safety state and everything it drives: the gate (ARMED), the external
// watchdog heartbeat, the fault log in flash and the FAULTS report. Every change of mode
// goes through here, so the outputs and the reported mode cannot disagree.
class SystemSupervisor {
 public:
  SystemSupervisor(ChannelController& channels, MeasurementService& measurement, OutputVerifier& outputs)
      : channels_(channels), measurement_(measurement), outputs_(outputs), loop_timing_(LOOP_BUDGET_US) {}

  // Loads the fault history kept in flash, counts the boot and records why the chip
  // last reset. A watchdog, panic or brownout reset is a critical fault: the board comes
  // up in Fault and will not arm until the operator has cleared it.
  void begin() {
    pinMode(HEARTBEAT_PIN, OUTPUT);
    digitalWrite(HEARTBEAT_PIN, LOW);

    reset_reason_ = esp_reset_reason();
    log_ready_ = log_.begin(FAULT_LOG_NAMESPACE, false);

    if (log_ready_) {
      state_.restore_history(log_.getULong("history", 0));
      persisted_history_ = state_.history();
      boots_ = log_.getULong("boots", 0);
      unexpected_resets_ = log_.getULong("unexpected", 0);
    }

    ++boots_;

    if (reset_was_unexpected(reset_reason_)) {
      ++unexpected_resets_;
      raise(safety::Fault::UnexpectedReset);
      resolve(safety::Fault::UnexpectedReset);
    }

    if (log_ready_) {
      log_.putULong("boots", boots_);
      log_.putULong("unexpected", unexpected_resets_);
    }

    persist_history();

    // Power-on self-test. Anything critical leaves the board in FAULT, so it cannot arm.
    self_test();
  }

  // The power-on self-test, and SELFTEST on demand. Checks what the outputs depend on:
  // the clocks the switch timing assumes, both switch generators, the ADC and memory.
  // Raises or resolves the matching faults and reports one SELFTEST line.
  void self_test() {
    const bool clocks_ok = clocks_as_designed();
    const bool switch_ok = channels_.switch_generators_ready();
    const bool adc_ok = !measurement_.lost();
    const bool memory_ok = memory_above_floor();

    set_fault(safety::Fault::ClockConfig, !clocks_ok);
    set_fault(safety::Fault::SwitchGenerator, !switch_ok);
    set_fault(safety::Fault::AdcLost, !adc_ok);
    set_fault(safety::Fault::LowMemory, !memory_ok);
    report_output_faults();

    // Output verification counts as passing when its hardware is not fitted ("off").
    const bool outputs_ok = strcmp(outputs_.gate_verdict(), "FAIL") != 0 && strcmp(outputs_.readback_verdict(), "FAIL") != 0;
    const bool pass = clocks_ok && switch_ok && adc_ok && memory_ok && outputs_ok;
    send_line(String("SELFTEST ") + (pass ? "PASS" : "FAIL") + " clocks=" + verdict(clocks_ok) +
                   " switch=" + verdict(switch_ok) + " adc=" + verdict(adc_ok) + " memory=" + verdict(memory_ok) +
                   " gate=" + outputs_.gate_verdict() + " readback=" + outputs_.readback_verdict() +
                   " mode=" + safety::name_of(state_.mode()));
  }

  // Continuous self-test, once per loop pass with that pass's duration. The ADC is
  // checked every pass so a lost one is reported promptly; memory and overruns once a
  // second, so a run of slow passes raises one fault rather than one per pass.
  void monitor(unsigned long now_ms, uint32_t pass_us) {
    loop_timing_.record(pass_us);
    set_fault(safety::Fault::AdcLost, measurement_.lost());

    outputs_.check(state_.armed());
    report_output_faults();

    if (now_ms - last_health_check_ms_ < HEALTH_CHECK_INTERVAL_MS) {
      return;
    }

    last_health_check_ms_ = now_ms;
    set_fault(safety::Fault::LowMemory, !memory_above_floor());

    if (loop_timing_.overruns() != overruns_at_last_check_) {
      overruns_at_last_check_ = loop_timing_.overruns();
      raise(safety::Fault::LoopOverrun);
      resolve(safety::Fault::LoopOverrun);
    }
  }

  void note_serial_overflow() {
    ++serial_overflows_;
    raise(safety::Fault::SerialOverflow);
    resolve(safety::Fault::SerialOverflow);
  }

  void note_bad_checksum() {
    ++bad_checksums_;
    raise(safety::Fault::BadChecksum);
    resolve(safety::Fault::BadChecksum);
  }

  // One line with everything needed to tell whether the board is keeping up: the worst
  // loop pass since the last report and since boot against its budget, memory headroom,
  // the ADC and the serial link.
  void report_health() {
    String line = String("HEALTH mode=") + safety::name_of(state_.mode());
    line += String(" uptime_ms=") + millis();
    line += String(" loop_max_us=") + loop_timing_.take_window_max();
    line += String(" loop_peak_us=") + loop_timing_.peak_us();
    line += String(" loop_budget_us=") + loop_timing_.budget_us();
    line += String(" overruns=") + loop_timing_.overruns();
    line += String(" heap_free=") + ESP.getFreeHeap();
    line += String(" heap_min=") + ESP.getMinFreeHeap();
    line += String(" stack_free=") + static_cast<uint32_t>(uxTaskGetStackHighWaterMark(nullptr));
    line += String(" adc=") + (measurement_.lost() ? "lost" : "ok");
    line += String(" adc_conversions=") + measurement_.conversions();
    line += String(" adc_timeouts=") + measurement_.timeouts();
    line += String(" rx_overflows=") + serial_overflows_;
    line += String(" bad_checksums=") + bad_checksums_;
    line += String(" gate=") + outputs_.gate_verdict();
    line += String(" gate_edges=") + outputs_.last_window_edges();
    line += String(" readback=") + outputs_.readback_verdict();
    send_line(line);
  }

  bool armed() const {
    return state_.armed();
  }

  // The checks that say this firmware image brought its own peripherals up. A missing
  // ADC or verification hardware is not the image's fault, so it does not count here.
  bool image_checks_passed() const {
    return (state_.active() & (safety::bit_of(safety::Fault::ClockConfig) | safety::bit_of(safety::Fault::SwitchGenerator))) == 0;
  }

  // ARM opens the gate. The outputs are zero whenever the board is not armed, so the gate
  // always opens onto a held-low switch line and zero setpoints.
  void arm() {
    const char* refusal = state_.arm_refusal();

    if (refusal != nullptr) {
      send_line(String("ERROR: Cannot arm - ") + refusal + "!");
      return;
    }

    state_.arm();
    channels_.set_switch_armed(true);
    send_line("ACK ARM");
  }

  void disarm() {
    make_safe();
    send_line("ACK DISARM");
  }

  // Zeroes every output and closes the gate, whatever the mode.
  void make_safe() {
    state_.disarm();
    channels_.engage_safe_state();
  }

  // A fault condition present now. Reported once at its onset; a critical one disarms.
  void raise(safety::Fault fault) {
    const bool onset = (state_.active() & safety::bit_of(fault)) == 0;

    if (state_.raise(fault)) {
      channels_.engage_safe_state();
    }

    if (onset) {
      const bool critical = safety::severity_of(fault) == safety::Severity::Critical;
      send_line(String("FAULT ") + safety::name_of(fault) + (critical ? " CRITICAL" : " WARNING") +
                     " mode=" + safety::name_of(state_.mode()));
    }

    persist_history();
  }

  void resolve(safety::Fault fault) {
    state_.resolve(fault);
  }

  void clear_faults() {
    state_.clear_latched();
    send_line(String("ACK CLEAR FAULTS mode=") + safety::name_of(state_.mode()));
  }

  void clear_log() {
    state_.clear_history();
    unexpected_resets_ = 0;

    if (log_ready_) {
      log_.putULong("unexpected", unexpected_resets_);
    }

    persist_history();
    send_line("ACK CLEAR LOG");
  }

  void report_faults() const {
    String line = String("FAULTS mode=") + safety::name_of(state_.mode());
    line += " active=" + fault_list(state_.active());
    line += " latched=" + fault_list(state_.latched());
    line += " history=" + fault_list(state_.history());
    line += " counts=" + fault_counts();
    line += String(" boots=") + boots_;
    line += String(" unexpected_resets=") + unexpected_resets_;
    line += String(" last_reset=") + reset_name(reset_reason_);
    send_line(line);
  }

  // Called once per loop pass, so the heartbeat stops if the loop does.
  void heartbeat() {
    heartbeat_high_ = !heartbeat_high_;
    digitalWrite(HEARTBEAT_PIN, heartbeat_high_ ? HIGH : LOW);
  }

 private:
  ChannelController& channels_;
  MeasurementService& measurement_;
  OutputVerifier& outputs_;
  safety::SafetyState state_;
  health::LoopTiming loop_timing_;
  unsigned long last_health_check_ms_ = 0;
  uint32_t overruns_at_last_check_ = 0;
  uint32_t serial_overflows_ = 0;
  uint32_t bad_checksums_ = 0;
  Preferences log_;
  bool log_ready_ = false;
  uint32_t persisted_history_ = 0;
  uint32_t boots_ = 0;
  uint32_t unexpected_resets_ = 0;
  esp_reset_reason_t reset_reason_ = ESP_RST_UNKNOWN;
  bool heartbeat_high_ = false;

  // The output checks' verdicts as faults. All three are critical: an output that is not
  // doing what it was told disarms the board.
  void report_output_faults() {
    set_fault(safety::Fault::GateMismatch, outputs_.gate_fault());
    set_fault(safety::Fault::SwitchFrequency, outputs_.frequency_fault());
    set_fault(safety::Fault::SetpointMismatch, outputs_.readback_fault());
  }

  void set_fault(safety::Fault fault, bool present) {
    if (present) {
      raise(fault);
    } else {
      resolve(fault);
    }
  }

  // The switch timing assumes APB at 80 MHz and REF_TICK at APB / 80 (switch_timing.h).
  static bool clocks_as_designed() {
    const uint32_t ref_tick_divider = REG_GET_FIELD(APB_CTRL_PLL_TICK_CONF_REG, APB_CTRL_PLL_TICK_NUM);
    return getApbFrequency() == APB_CLK_FREQ && ref_tick_divider + 1 == APB_CLK_FREQ / SWITCH_LEDC_CLOCK_HZ;
  }

  // Free heap now, and the loop task's stack at its deepest so far (that one cannot recover,
  // so a stack that once came close stays reported).
  static bool memory_above_floor() {
    return ESP.getFreeHeap() >= HEAP_FLOOR_BYTES && uxTaskGetStackHighWaterMark(nullptr) >= STACK_FLOOR_BYTES;
  }

  static const char* verdict(bool ok) {
    return ok ? "ok" : "FAIL";
  }

  // Flash is only written when the history gains a fault, so at most once per fault type
  // between CLEAR LOGs, never at loop rate.
  void persist_history() {
    if (!log_ready_ || state_.history() == persisted_history_) {
      return;
    }

    log_.putULong("history", state_.history());
    persisted_history_ = state_.history();
  }

  static bool reset_was_unexpected(esp_reset_reason_t reason) {
    return reason == ESP_RST_PANIC || reason == ESP_RST_INT_WDT || reason == ESP_RST_TASK_WDT ||
           reason == ESP_RST_WDT || reason == ESP_RST_BROWNOUT;
  }

  static const char* reset_name(esp_reset_reason_t reason) {
    switch (reason) {
      case ESP_RST_POWERON: return "POWERON";
      case ESP_RST_EXT: return "EXTERNAL";
      case ESP_RST_SW: return "SOFTWARE";
      case ESP_RST_PANIC: return "PANIC";
      case ESP_RST_INT_WDT: return "INT_WDT";
      case ESP_RST_TASK_WDT: return "TASK_WDT";
      case ESP_RST_WDT: return "WDT";
      case ESP_RST_DEEPSLEEP: return "DEEPSLEEP";
      case ESP_RST_BROWNOUT: return "BROWNOUT";
      case ESP_RST_SDIO: return "SDIO";
      default: return "UNKNOWN";
    }
  }

  static String fault_list(uint32_t faults) {
    String list;

    for (uint8_t index = 0; index < safety::FAULT_COUNT; ++index) {
      if (faults & (1UL << index)) {
        if (list.length() > 0) {
          list += ",";
        }

        list += safety::name_of(static_cast<safety::Fault>(index));
      }
    }

    return list.length() > 0 ? list : String("none");
  }

  String fault_counts() const {
    String list;

    for (uint8_t index = 0; index < safety::FAULT_COUNT; ++index) {
      const safety::Fault fault = static_cast<safety::Fault>(index);

      if (state_.count(fault) > 0) {
        if (list.length() > 0) {
          list += ",";
        }

        list += String(safety::name_of(fault)) + ":" + state_.count(fault);
      }
    }

    return list.length() > 0 ? list : String("none");
  }
};

class CommandProcessor {
 public:
  CommandProcessor(ChannelController& channels, MeasurementService& measurement, LedIndicator& leds,
                   SystemSupervisor& supervisor)
      : channels_(channels), measurement_(measurement), leds_(leds), supervisor_(supervisor) {}

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
          supervisor_.note_serial_overflow();
        }
      }
    }
  }

 private:
  ChannelController& channels_;
  MeasurementService& measurement_;
  LedIndicator& leds_;
  SystemSupervisor& supervisor_;
  String command_buffer_;
  unsigned long last_command_ms_ = 0;
  bool command_seen_ = false;

  // Which firmware this is, which protocol it speaks, and which verification hardware the
  // build expects. The backend checks protocol= before it will arm the board.
  static void report_version() {
    send_line(String("VERSION firmware=") + TESTBED_FIRMWARE_VERSION + " protocol=" + PROTOCOL_VERSION +
              " build=" + TESTBED_BUILD_ENV + " gate_loopback=" + (GATE_LOOPBACK_FITTED ? 1 : 0) +
              " setpoint_readback=" + (SETPOINT_READBACK_FITTED ? 1 : 0));
  }

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

    // A line whose CRC does not match is refused whole and does not count as the host
    // being alive: a link that only delivers garbage must still trip the failsafe.
    size_t body_length = 0;

    if (line_protocol::verify(command.c_str(), command.length(), &body_length) == line_protocol::Check::Invalid) {
      send_line("ERROR: Bad checksum!");
      supervisor_.note_bad_checksum();
      return;
    }

    command.remove(body_length);
    command.trim();

    if (command.isEmpty()) {
      return;
    }

    last_command_ms_ = millis();
    command_seen_ = true;

    // Commands that would drive an output are refused unless the board is armed; ones that
    // set an output to zero are always accepted.
    if (command.equalsIgnoreCase("PING")) {
      send_line("OK");
    } else if (command.equalsIgnoreCase("VERSION")) {
      report_version();
    } else if (command.equalsIgnoreCase("ARM")) {
      supervisor_.arm();
    } else if (command.equalsIgnoreCase("DISARM")) {
      supervisor_.disarm();
    } else if (command.equalsIgnoreCase("FAULTS")) {
      supervisor_.report_faults();
    } else if (command.equalsIgnoreCase("CLEAR FAULTS")) {
      supervisor_.clear_faults();
    } else if (command.equalsIgnoreCase("CLEAR LOG")) {
      supervisor_.clear_log();
    } else if (command.equalsIgnoreCase("SELFTEST")) {
      supervisor_.self_test();
    } else if (command.equalsIgnoreCase("HEALTH")) {
      supervisor_.report_health();
    } else if (command.equalsIgnoreCase("READ")) {
      measurement_.request_reading();
      leds_.blink_once(80);
    } else if (command.equalsIgnoreCase("GET PINS") || command.equalsIgnoreCase("PINS")) {
      channels_.report_channels();
    } else if (command.startsWith("TARGETS")) {
      const int first_space_index = command.indexOf(' ');

      if (first_space_index <= 0) {
        send_line("ERROR: TARGETS requires five voltages!");
        return;
      }

      const String args = command.substring(first_space_index + 1);

      if (!supervisor_.armed() && ChannelController::targets_would_energise(args)) {
        send_line("ERROR: Not armed!");
        return;
      }

      if (channels_.apply_target_voltages(args)) {
        send_line("ACK TARGETS");
      } else {
        send_line("ERROR: TARGETS requires five voltages!");
      }
    } else if (command.startsWith("PIN")) {
      const int first_space_index = command.indexOf(' ');
      const int second_space_index = command.indexOf(' ', first_space_index + 1);

      if (first_space_index > 0 && second_space_index > first_space_index) {
        const String token = command.substring(first_space_index + 1, second_space_index);
        const uint8_t pin_index = to_pin_index(token);

        const int value = command.substring(second_space_index + 1).toInt();

        if (!supervisor_.armed() && value > 0) {
          send_line("ERROR: Not armed!");
          return;
        }

        channels_.set_channel(pin_index, value);
      } else {
        send_line("ERROR: Invalid PIN syntax!");
      }
    } else if (command.startsWith("SWITCH_PERIOD_US")) {
      const int first_space_index = command.indexOf(' ');

      if (first_space_index <= 0) {
        send_line("ERROR: Invalid switch time!");
        return;
      }

      String value_token = command.substring(first_space_index + 1);
      value_token.trim();
      long switch_period_us = value_token.toInt();

      if (switch_period_us <= 0) {
        channels_.stop_switching(0);
        send_line("ACK SWITCH_PERIOD_US 0 (disabled)");
        return;
      }

      if (!supervisor_.armed()) {
        send_line("ERROR: Not armed!");
        return;
      }

      if (switch_period_us < SWITCH_PERIOD_MIN_US || switch_period_us > SWITCH_PERIOD_MAX_US) {
        send_line("ERROR: Switch time out of bounds!");
        return;
      }

      if (!channels_.automate_switching(static_cast<unsigned long>(switch_period_us))) {
        send_line("ERROR: Switch timer unavailable!");
        return;
      }

      send_line(String("ACK SWITCH_PERIOD_US ") + switch_period_us);
    } else {
      send_line("ERROR: Unknown command received!");
      leds_.blink_error(2, 70);
    }
  }
};

ChannelController channels;
MeasurementService measurement_service(ads);
LedIndicator led_indicator(LED_PIN);
OtaWifiService ota_wifi_service;
OutputVerifier output_verifier(channels, measurement_service);
SystemSupervisor supervisor(channels, measurement_service, output_verifier);
CommandProcessor command_processor(channels, measurement_service, led_indicator, supervisor);

bool failsafe_engaged = false;

// The core asks this before marking a freshly updated image as good (esp32-hal-misc.c).
// Saying "later" leaves the image pending until setup() has run the power-on self-test.
bool verifyRollbackLater() {
  return true;
}

// After an OTA update the bootloader runs the new image once, pending verification. It is
// kept only if its self-test shows it brought up its own clocks and switch generators;
// otherwise the bootloader goes back to the previous image. An image flashed over the
// cable is never pending, so this does nothing then.
void confirm_or_roll_back_image() {
  const esp_partition_t* running = esp_ota_get_running_partition();
  esp_ota_img_states_t state = ESP_OTA_IMG_UNDEFINED;

  if (running == nullptr || esp_ota_get_state_partition(running, &state) != ESP_OK || state != ESP_OTA_IMG_PENDING_VERIFY) {
    return;
  }

  if (supervisor.image_checks_passed()) {
    esp_ota_mark_app_valid_cancel_rollback();
    send_line("OTA image confirmed by the power-on self-test");
  } else {
    send_line("ERROR: OTA image failed its power-on self-test, rolling back!");
    Serial.flush();
    esp_ota_mark_app_invalid_rollback_and_reboot();
  }
}

// Once the host has spoken to us it is expected to keep doing so. If it goes quiet the
// board disarms, otherwise a set of targets would stay latched on the rig for as long as
// it has power. The host coming back does not re-arm it: that takes an explicit ARM.
void enforce_command_timeout(unsigned long now_ms) {
  if (!command_processor.has_received_command()) {
    return;
  }

  const bool host_overdue = (now_ms - command_processor.last_command_ms()) >= COMMAND_TIMEOUT_MS;

  if (host_overdue && !failsafe_engaged) {
    supervisor.make_safe();
    failsafe_engaged = true;
    send_line("FAILSAFE outputs zeroed, no command received from host");
    supervisor.raise(safety::Fault::HostTimeout);

  } else if (!host_overdue && failsafe_engaged) {
    failsafe_engaged = false;
    supervisor.resolve(safety::Fault::HostTimeout);
    send_line("FAILSAFE cleared, host link restored");
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
  output_verifier.begin();

  // Records the reset and runs the power-on self-test, so after the ADC, the switch
  // generators and the output checks have been brought up.
  supervisor.begin();
  confirm_or_roll_back_image();

  if (WIRELESS_ENABLED) {
    ota_wifi_service.begin(WIFI_SSID, WIFI_PASSWORD, OTA_HOSTNAME);
  }

  // The board comes up Safe, with the gate closed, and stays that way until an ARM.
  // From here a loop that stops for 5 s resets the chip, which records the reset as a
  // fault and brings the board back up Safe.
  enableLoopWDT();

  LOG_INFO("Setup complete. Awaiting commands...");
  led_indicator.set_low();
}

void loop() {
  const unsigned long pass_started_us = micros();
  supervisor.heartbeat();
  command_processor.poll_serial();

  const unsigned long now_ms = millis();
  measurement_service.update(now_ms);
  led_indicator.heartbeat(now_ms);
  led_indicator.update(now_ms);
  if (WIRELESS_ENABLED) {
    ota_wifi_service.loop(!supervisor.armed());
  }
  enforce_command_timeout(now_ms);
  supervisor.monitor(now_ms, static_cast<uint32_t>(micros() - pass_started_us));

  delay(1);
}
