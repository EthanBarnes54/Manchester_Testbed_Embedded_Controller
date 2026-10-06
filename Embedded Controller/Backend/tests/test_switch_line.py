"""The firmware's switch line state machine, compiled from main.cpp and run on the host.

The real SwitchLine class and the real switch constants are cut out of main.cpp and built
against stand-ins for the GPIO matrix, the LEDC peripheral and the hardware timer that
record every operation. That shows the order things happen in on each transition, which
is what decides whether the pin can glitch, without a board.
"""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
FIRMWARE_DIR = BACKEND_DIR.parent
INCLUDE_DIR = FIRMWARE_DIR / "include"
MAIN_CPP = (FIRMWARE_DIR / "src" / "main.cpp").read_text(encoding="utf-8")


def firmware_source():
    """The constants SwitchLine depends on and the class itself, verbatim from main.cpp."""

    lines = [
        re.search(rf"^constexpr int {name} = [^;]+;", MAIN_CPP, re.M).group(0)
        for name in ("SWITCH_LOGIC_PIN", "CONTROLLED_PULSE_CHANNELS")
    ]
    lines.append(re.search(r"^constexpr int LED_CONTROL_CHANNELS\[[^;]+;", MAIN_CPP, re.M).group(0))
    lines.append(
        re.search(
            r"^constexpr int SWITCH_HARDWARE_MAX_US.*?^constexpr unsigned long SWITCH_LEDC_SETTLE_MARGIN_US[^\n]*$",
            MAIN_CPP,
            re.S | re.M,
        ).group(0)
    )
    lines.append(re.search(r"^class SwitchLine \{.*?^\};", MAIN_CPP, re.S | re.M).group(0))
    lines.append("SwitchLine* SwitchLine::instance_ = nullptr;")
    return "\n".join(lines)


HARNESS = r"""
#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>
#include "switch_timing.h"

// ---- Stand-ins for the Arduino core and the IDF, recording what the class does ----

static std::vector<std::string> ops;
static unsigned long now_us = 0;
static int critical_depth = 0;
static void op(const std::string& what) { ops.push_back(what + (critical_depth ? " [critical]" : "")); }

#define IRAM_ATTR
#define OUTPUT 1
#define BIT(nr) (1UL << (nr))
#define REF_CLK_FREQ (1000000)
typedef int esp_err_t;
#define ESP_OK 0
typedef int portMUX_TYPE;
#define portMUX_INITIALIZER_UNLOCKED 0
#define portENTER_CRITICAL(mux) (++critical_depth)
#define portEXIT_CRITICAL(mux) (--critical_depth)
#define portENTER_CRITICAL_ISR(mux) (++critical_depth)
#define portEXIT_CRITICAL_ISR(mux) (--critical_depth)
#define LOG_INFO(x)
#define LOG_ERROR(x) op("log_error")

enum ledc_mode_t { LEDC_HIGH_SPEED_MODE = 0, LEDC_LOW_SPEED_MODE };
enum ledc_timer_t { LEDC_TIMER_0, LEDC_TIMER_1, LEDC_TIMER_2, LEDC_TIMER_3 };
enum ledc_channel_t { LEDC_CHANNEL_0 };
enum ledc_clk_src_t { LEDC_APB_CLK = 1, LEDC_REF_TICK = 3 };
constexpr uint32_t SIG_GPIO_OUT_IDX = 256;
constexpr uint32_t LEDC_HS_SIG_OUT0_IDX = 71;
constexpr uint32_t GPIO_FUNC0_OEN_SEL = BIT(10);

static uint32_t gpio_out = 0;
static uint32_t pin_route[40] = {0};
static bool pin_output_enabled = false;

struct SetReg { SetReg& operator=(uint32_t mask) { gpio_out |= mask; op("gpio_set"); return *this; } };
struct ClearReg { ClearReg& operator=(uint32_t mask) { gpio_out &= ~mask; op("gpio_clear"); return *this; } };
struct RouteReg {
  int pin = 0;
  RouteReg& operator=(uint32_t value) {
    pin_route[pin] = value;
    op((value & 0x1FF) == SIG_GPIO_OUT_IDX ? "route_gpio" : "route_ledc " + std::to_string(value & 0x1FF));
    return *this;
  }
};
struct RouteCfg { RouteReg val; };
struct GpioDev {
  SetReg out_w1ts;
  ClearReg out_w1tc;
  RouteCfg func_out_sel_cfg[40];
  GpioDev() { for (int i = 0; i < 40; ++i) func_out_sel_cfg[i].val.pin = i; }
} GPIO;

void pinMode(int, int) { pin_output_enabled = true; op("pinMode"); }

// LEDC: one timer's configuration, its run state, and the overflow flag the class polls.
struct TimerConf { uint32_t clock_divider = 0, duty_resolution = 0, tick_sel = 1; };
struct TimerRegs { TimerConf conf; };
struct TimerGroup { TimerRegs timer[4]; };
struct ClearIntReg { ClearIntReg& operator=(uint32_t value); };
struct LedcDev {
  TimerGroup timer_group[2];
  struct { uint32_t val = 0; } int_raw;
  struct { ClearIntReg val; } int_clr;
} LEDC;
ClearIntReg& ClearIntReg::operator=(uint32_t value) { LEDC.int_raw.val &= ~value; op("clear_overflow"); return *this; }

static bool ledc_paused = true;
static unsigned long ledc_count_started_us = 0;
static bool ledc_never_overflows = false;
static bool ledc_drops_fraction = false;
static bool ledc_bind_fails = false;

esp_err_t ledc_timer_set(ledc_mode_t mode, ledc_timer_t timer, uint32_t divider, uint32_t bits, ledc_clk_src_t src) {
  TimerConf& conf = LEDC.timer_group[mode].timer[timer].conf;
  conf.clock_divider = (ledc_drops_fraction ? divider + 1 : divider) & 0x3FFFF;
  conf.duty_resolution = bits & 0x1F;
  conf.tick_sel = (src == LEDC_APB_CLK);
  op("timer_set " + std::to_string(divider) + " " + std::to_string(bits) + (src == LEDC_REF_TICK ? " ref" : " apb"));
  return ESP_OK;
}
esp_err_t ledc_timer_pause(ledc_mode_t, ledc_timer_t) { ledc_paused = true; op("ledc_pause"); return ESP_OK; }
esp_err_t ledc_timer_resume(ledc_mode_t, ledc_timer_t) { ledc_paused = false; op("ledc_resume"); return ESP_OK; }
esp_err_t ledc_timer_rst(ledc_mode_t, ledc_timer_t) { ledc_count_started_us = now_us; op("ledc_rst"); return ESP_OK; }
esp_err_t ledc_bind_channel_timer(ledc_mode_t, ledc_channel_t, ledc_timer_t) { op("ledc_bind"); return ledc_bind_fails ? -1 : ESP_OK; }
esp_err_t ledc_set_duty_with_hpoint(ledc_mode_t, ledc_channel_t, uint32_t duty, uint32_t hpoint) {
  op("ledc_duty " + std::to_string(duty) + " hpoint " + std::to_string(hpoint));
  return ESP_OK;
}
esp_err_t ledc_update_duty(ledc_mode_t, ledc_channel_t) { op("ledc_update"); return ESP_OK; }

// Each call is one microsecond. A running LEDC timer overflows once a full cycle
// (divider x 2^bits ticks of the 1 MHz REF_TICK) has passed since it was last reset.
unsigned long micros() {
  ++now_us;
  const TimerConf& conf = LEDC.timer_group[0].timer[3].conf;
  const unsigned long cycle_us = (conf.clock_divider >> 8) << conf.duty_resolution;
  if (!ledc_paused && !ledc_never_overflows && now_us - ledc_count_started_us >= cycle_us) LEDC.int_raw.val |= BIT(3);
  return now_us;
}

struct hw_timer_t { uint64_t count = 0, alarm = 0; bool autoreload = false, alarm_enabled = false; void (*isr)() = nullptr; };
static hw_timer_t hardware_timer;
static bool timer_available = true;
hw_timer_t* timerBegin(uint8_t, uint16_t, bool) { return timer_available ? &hardware_timer : nullptr; }
void timerAttachInterrupt(hw_timer_t* t, void (*fn)(), bool) { t->isr = fn; }
void timerAlarmDisable(hw_timer_t* t) { t->alarm_enabled = false; op("alarm_disable"); }
void timerAlarmEnable(hw_timer_t* t) { t->alarm_enabled = true; op("alarm_enable"); }
void timerAlarmWrite(hw_timer_t* t, uint64_t value, bool autoreload) {
  t->alarm = value; t->autoreload = autoreload; op("alarm_write " + std::to_string(value) + (autoreload ? " reload" : ""));
}
void timerWrite(hw_timer_t* t, uint64_t value) { t->count = value; op("timer_write " + std::to_string(value)); }

// ---- The firmware under test ----

@@FIRMWARE@@

// ---- Scenarios ----

static int failures = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++failures; } } while (0)

static const uint32_t pin = SWITCH_LOGIC_PIN;
static bool on_gpio() { return (pin_route[pin] & 0x1FF) == SIG_GPIO_OUT_IDX && (pin_route[pin] & GPIO_FUNC0_OEN_SEL); }
static bool on_ledc() { return (pin_route[pin] & 0x1FF) == SWITCH_LEDC_SIGNAL && (pin_route[pin] & GPIO_FUNC0_OEN_SEL); }
static bool gpio_high() { return (gpio_out >> pin) & 1; }

static int index_of(const std::string& what, int from = 0) {
  for (int i = from; i < static_cast<int>(ops.size()); ++i) if (ops[i].compare(0, what.size(), what) == 0) return i;
  return -1;
}
static int count_of(const std::string& what) {
  int n = 0;
  for (const std::string& o : ops) if (o.compare(0, what.size(), what) == 0) ++n;
  return n;
}
static void fire_timer_interrupt() { if (hardware_timer.isr) hardware_timer.isr(); }

static void reset_world() {
  ops.clear(); now_us = 1000; critical_depth = 0; gpio_out = BIT(pin); pin_output_enabled = false;
  for (uint32_t& r : pin_route) r = 0;
  LEDC = LedcDev(); ledc_paused = true; ledc_count_started_us = 0;
  ledc_never_overflows = ledc_drops_fraction = ledc_bind_fails = false;
  hardware_timer = hw_timer_t(); timer_available = true;
}

static void dump() { for (const std::string& o : ops) std::printf("    %s\n", o.c_str()); }

int main() {
  { // Boot: the output register is cleared before the output is enabled.
    reset_world();
    SwitchLine::hold_low_at_boot();
    CHECK(ops.size() == 2 && ops[0] == "gpio_clear" && ops[1] == "pinMode");
    CHECK(!gpio_high() && pin_output_enabled);
  }

  { // begin(): held low on the output register, both generators parked.
    reset_world();
    SwitchLine line; line.begin();
    CHECK(on_gpio() && !gpio_high() && line.reported_level() == 0);
    CHECK(ledc_paused && !hardware_timer.alarm_enabled && index_of("ledc_bind") >= 0);
  }

  { // Hardware start at the floor and at the hardware ceiling.
    const unsigned long periods[] = {5, 1000};
    const char* settings[] = {"timer_set 1280 1 ref", "timer_set 32000 4 ref"};
    for (int k = 0; k < 2; ++k) {
      reset_world();
      SwitchLine line; line.begin(); ops.clear();
      const unsigned long held_at = now_us;
      CHECK(line.start(periods[k]));
      CHECK(on_ledc() && !ledc_paused && line.reported_level() == 1);

      const int held = index_of("route_gpio");
      const int programmed = index_of(settings[k]);
      const int duty = index_of("ledc_duty");
      const int priming = index_of("ledc_resume");
      const int restart = index_of("ledc_rst", priming);
      const int routed = index_of("route_ledc");
      const int released = index_of("ledc_resume", routed);
      CHECK(held >= 0 && held < programmed && programmed < duty && duty < priming);
      CHECK(priming < restart && restart < routed && routed < released);
      CHECK(count_of("route_ledc") == 1);
      CHECK(routed >= 0 && released >= 0);
      if (routed >= 0 && released >= 0) {
        CHECK(ops[routed].find("[critical]") != std::string::npos && ops[released].find("[critical]") != std::string::npos);
      }
      // The line was held low for at least one whole new cycle before the pin was handed over.
      CHECK(now_us - held_at >= 2 * periods[k]);
      if (failures) dump();
    }
  }

  { // Duty is exactly half the counter: 5 us is 1 bit, 1000 us is 4 bits.
    reset_world();
    SwitchLine line; line.begin(); ops.clear();
    line.start(5);
    CHECK(index_of("ledc_duty 1 hpoint 0") >= 0);
    line.start(1000);
    CHECK(index_of("ledc_duty 8 hpoint 0") >= 0);
  }

  { // Above the hardware ceiling the timer interrupt takes over, counting from zero.
    reset_world();
    SwitchLine line; line.begin(); ops.clear();
    CHECK(line.start(SWITCH_HARDWARE_MAX_US + 1));
    CHECK(on_gpio() && !gpio_high() && line.reported_level() == 1 && ledc_paused);
    const int held = index_of("route_gpio"), zeroed = index_of("timer_write 0");
    const int armed = index_of("alarm_write " + std::to_string(SWITCH_HARDWARE_MAX_US + 1) + " reload");
    const int enabled = index_of("alarm_enable");
    CHECK(held >= 0 && held < zeroed && zeroed < armed && armed < enabled);
    fire_timer_interrupt(); CHECK(gpio_high());
    fire_timer_interrupt(); CHECK(!gpio_high());
  }

  { // Failsafe from interrupt switching: an interrupt already pending cannot raise the line.
    reset_world();
    SwitchLine line; line.begin();
    line.start(5000);
    fire_timer_interrupt(); CHECK(gpio_high());
    ops.clear();
    line.hold(false);
    CHECK(!gpio_high() && on_gpio() && !hardware_timer.alarm_enabled && line.reported_level() == 0);
    CHECK(index_of("gpio_clear") < index_of("alarm_disable"));
    // Twice, so a handler that ignored the mode would raise the line on one of them.
    fire_timer_interrupt();
    CHECK(!gpio_high() && line.reported_level() == 0);
    fire_timer_interrupt();
    CHECK(!gpio_high() && line.reported_level() == 0);
  }

  { // Failsafe from hardware switching: the pin is back on a low output register before LEDC stops.
    reset_world();
    SwitchLine line; line.begin();
    line.start(5);
    gpio_out |= BIT(pin);  // a stale high left in the output register must not reach the pin
    ops.clear();
    line.hold(false);
    CHECK(on_gpio() && !gpio_high() && ledc_paused && line.reported_level() == 0);
    CHECK(index_of("gpio_clear") < index_of("route_gpio") && index_of("route_gpio") < index_of("ledc_pause"));
  }

  { // A fixed level, and the timer interrupt does nothing while the line is held.
    reset_world();
    SwitchLine line; line.begin();
    line.hold(true);
    CHECK(on_gpio() && gpio_high() && line.reported_level() == 1);
    fire_timer_interrupt();
    CHECK(gpio_high());
    fire_timer_interrupt();
    CHECK(gpio_high());
    line.hold(false);
    CHECK(!gpio_high() && line.reported_level() == 0);
  }

  { // Changing the period while running drops the line low before anything is reprogrammed.
    reset_world();
    SwitchLine line; line.begin();
    line.start(5);
    ops.clear();
    CHECK(line.start(7));
    CHECK(index_of("route_gpio") >= 0 && index_of("route_gpio") < index_of("timer_set 1792 1 ref"));
    CHECK(on_ledc() && line.reported_level() == 1);
  }

  { // The LEDC never completing a cycle is refused, with the line left low on the output register.
    reset_world();
    SwitchLine line; line.begin();
    ledc_never_overflows = true;
    ops.clear();
    CHECK(!line.start(5));
    CHECK(on_gpio() && !gpio_high() && ledc_paused && line.reported_level() == 0);
    CHECK(count_of("route_ledc") == 0 && index_of("log_error") >= 0);
  }

  { // A divider the hardware did not take exactly is refused rather than run.
    reset_world();
    SwitchLine line; line.begin();
    ledc_drops_fraction = true;
    ops.clear();
    CHECK(!line.start(5));
    CHECK(on_gpio() && !gpio_high() && count_of("route_ledc") == 0 && count_of("ledc_resume") == 0);
  }

  { // With a generator missing, the request fails and nothing about the line changes.
    reset_world();
    timer_available = false;
    SwitchLine line; line.begin();
    line.hold(true);
    CHECK(!line.start(SWITCH_HARDWARE_MAX_US + 1));
    CHECK(gpio_high() && line.reported_level() == 1);

    reset_world();
    ledc_bind_fails = true;
    SwitchLine other; other.begin();
    CHECK(!other.start(5));
    CHECK(on_gpio() && !gpio_high() && other.reported_level() == 0);
  }

  CHECK(critical_depth == 0);
  return failures ? 1 : 0;
}
"""


@pytest.mark.skipif(shutil.which("g++") is None, reason="needs a host C++ compiler")
def test_switch_line_transitions_on_the_host(tmp_path):
    source = tmp_path / "switch_line.cpp"
    binary = tmp_path / "switch_line"
    source.write_text(HARNESS.replace("@@FIRMWARE@@", firmware_source()))

    build = subprocess.run(
        ["g++", "-std=gnu++11", "-Wall", f"-I{INCLUDE_DIR}", "-o", str(binary), str(source)],
        capture_output=True,
        text=True,
    )
    assert build.returncode == 0, build.stderr

    run = subprocess.run([str(binary)], capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stdout + run.stderr
