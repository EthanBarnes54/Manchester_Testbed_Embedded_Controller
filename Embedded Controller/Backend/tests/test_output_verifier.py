"""The firmware's output verification, compiled from main.cpp and run on the host.

The real OutputVerifier class and its constants are cut out of main.cpp and built against a
fake pulse counter, gate input, switch line and ADC readback, once with the verification
hardware declared fitted and once without it.
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
    parts = [re.search(rf"^constexpr int {name} = [^;]+;", MAIN_CPP, re.M).group(0)
             for name in ("MODULATION_RESOLUTION", "MAX_MODULATION_VALUE", "GATE_LOOPBACK_PIN")]
    parts.append(re.search(r"^#ifndef TESTBED_GATE_LOOPBACK.*?^constexpr uint8_t READBACK_MISMATCH_READINGS = [^;]+;",
                           MAIN_CPP, re.S | re.M).group(0))
    parts.append(re.search(r"^class OutputVerifier \{.*?^\};", MAIN_CPP, re.S | re.M).group(0))
    return "\n".join(parts)


HARNESS = r"""
#include <cstdio>
#include <cstdint>
#include "output_checks.h"

// ---- Stand-ins: pulse counter, GPIO, time ----
typedef int esp_err_t;
#define ESP_OK 0
#define INPUT 1
#define HIGH 1
#define LOW 0
enum pcnt_unit_t { PCNT_UNIT_0 };
enum pcnt_channel_t { PCNT_CHANNEL_0 };
enum { PCNT_MODE_KEEP, PCNT_COUNT_INC, PCNT_COUNT_DIS };
#define PCNT_PIN_NOT_USED (-1)
struct pcnt_config_t { int pulse_gpio_num, ctrl_gpio_num, lctrl_mode, hctrl_mode, pos_mode, neg_mode; int16_t counter_h_lim, counter_l_lim; pcnt_unit_t unit; pcnt_channel_t channel; };

static unsigned long now_us = 0;
static int16_t pcnt_count = 0;
static bool pcnt_config_fails = false;
static int configured_pin = -1, configured_pos = -1, configured_filter = -1;
static int gate_level = LOW;

unsigned long micros() { return now_us; }
void pinMode(int, int) {}
int digitalRead(int) { return gate_level; }
esp_err_t pcnt_unit_config(const pcnt_config_t* c) { configured_pin = c->pulse_gpio_num; configured_pos = c->pos_mode; return pcnt_config_fails ? -1 : ESP_OK; }
esp_err_t pcnt_set_filter_value(pcnt_unit_t, uint16_t value) { configured_filter = value; return ESP_OK; }
esp_err_t pcnt_filter_enable(pcnt_unit_t) { return ESP_OK; }
esp_err_t pcnt_counter_clear(pcnt_unit_t) { pcnt_count = 0; return ESP_OK; }
esp_err_t pcnt_get_counter_value(pcnt_unit_t, int16_t* value) { *value = pcnt_count; return ESP_OK; }

// ---- Stand-ins: the switch line, the setpoints and the ADC readback ----
struct SwitchLine { struct Snapshot { bool switching; bool held_high; uint32_t period_us; uint32_t generation; }; };

struct ChannelController {
  SwitchLine::Snapshot line = {false, false, 0, 1};
  int values[5] = {0};
  unsigned long changed_ms[5] = {0};
  SwitchLine::Snapshot switch_snapshot() const { return line; }
  int setpoint_value(uint8_t n) const { return values[n - 1]; }
  unsigned long setpoint_changed_ms(uint8_t n) const { return changed_ms[n - 1]; }
};

struct MeasurementService {
  float volts[4] = {0};
  unsigned long at_ms[4] = {0};
  float readback_volts(uint8_t input) const { return volts[input]; }
  unsigned long readback_ms(uint8_t input) const { return at_ms[input]; }
};

// ---- The firmware under test ----
@@FIRMWARE@@

// ---- Scenarios ----
static int failures = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++failures; } } while (0)

// Runs passes of 1 ms. While the line is switching and the gate is open, the counter sees
// the edges a square wave with that period would put through, scaled by edge_scale.
static void run_ms(OutputVerifier& v, ChannelController& c, bool armed, unsigned ms, double edge_scale = 1.0) {
  static double carry = 0;
  for (unsigned i = 0; i < ms; ++i) {
    now_us += 1000;
    if (armed && c.line.switching) {
      carry += edge_scale * 1000.0 / (2.0 * c.line.period_us);
      const int whole = static_cast<int>(carry);
      carry -= whole;
      pcnt_count = static_cast<int16_t>(pcnt_count + whole);
    }
    v.check(armed);
  }
}

static void reset() { now_us = 1000000; pcnt_count = 0; pcnt_config_fails = false; gate_level = LOW; }

int main() {
#if TESTBED_GATE_LOOPBACK
  { // The counter is set up on the loopback pin, counting rising edges through a short filter.
    reset(); ChannelController c; MeasurementService m; OutputVerifier v(c, m); v.begin();
    CHECK(configured_pin == GATE_LOOPBACK_PIN && configured_pos == PCNT_COUNT_INC && configured_filter == LOOPBACK_FILTER_APB_CYCLES);
    run_ms(v, c, false, 100);
    CHECK(!v.gate_fault() && !v.frequency_fault() && std::string(v.gate_verdict()) == "ok");
  }

  { // Disarmed, a gate output that is high is a fault, but only after several passes in a row.
    reset(); ChannelController c; MeasurementService m; OutputVerifier v(c, m); v.begin();
    gate_level = HIGH;
    run_ms(v, c, false, GATE_MISMATCH_PASSES - 1);
    CHECK(!v.gate_fault());
    run_ms(v, c, false, 1);
    CHECK(v.gate_fault() && std::string(v.gate_verdict()) == "FAIL");
    gate_level = LOW;
    run_ms(v, c, false, 1);
    CHECK(!v.gate_fault());
  }

  { // Armed and held, the gate output must follow the held level.
    reset(); ChannelController c; MeasurementService m; OutputVerifier v(c, m); v.begin();
    c.line = {false, true, 0, 2};
    gate_level = HIGH;
    run_ms(v, c, true, 50);
    CHECK(!v.gate_fault());
    gate_level = LOW;
    run_ms(v, c, true, 5);
    CHECK(v.gate_fault());
  }

  { // Switching at 5 us and at the 1 us floor: the right edge count passes, half of it does not.
    const uint32_t periods[] = {1, 5, 1000};
    for (uint32_t period : periods) {
      reset(); ChannelController c; MeasurementService m; OutputVerifier v(c, m); v.begin();
      c.line = {true, false, period, 3};
      run_ms(v, c, true, 100);                       // the first window spans arming and is thrown away
      CHECK(!v.frequency_fault() && !v.gate_fault());
      CHECK(v.last_window_edges() > 0);
      run_ms(v, c, true, 100, 0.5);
      CHECK(v.frequency_fault());
      run_ms(v, c, true, 100);
      CHECK(!v.frequency_fault());
    }
  }

  { // A window that spans a period change, or arming, is thrown away rather than judged.
    reset(); ChannelController c; MeasurementService m; OutputVerifier v(c, m); v.begin();
    run_ms(v, c, false, 10);                         // arm half way through a window
    c.line = {true, false, 5, 4};
    run_ms(v, c, true, 60);                          // and change period half way through another
    CHECK(!v.frequency_fault());
    c.line = {true, false, 50, 5};
    run_ms(v, c, true, 20);                          // this window holds 10 ms of each period
    CHECK(!v.frequency_fault());
    run_ms(v, c, true, 60);
    CHECK(!v.frequency_fault());
  }

  { // Edges through a closed gate are a gate fault.
    reset(); ChannelController c; MeasurementService m; OutputVerifier v(c, m); v.begin();
    run_ms(v, c, false, 25);
    pcnt_count = 40;
    run_ms(v, c, false, 25);
    CHECK(v.gate_fault());
  }

  { // A counter that will not start is a fault: the check cannot be trusted.
    reset(); pcnt_config_fails = true; ChannelController c; MeasurementService m; OutputVerifier v(c, m); v.begin();
    CHECK(v.gate_fault());
  }
#else
  { // Not fitted: nothing is configured, and nothing is ever reported, whatever the pin does.
    reset(); configured_pin = -1; ChannelController c; MeasurementService m; OutputVerifier v(c, m); v.begin();
    gate_level = HIGH; pcnt_count = 500;
    run_ms(v, c, false, 200);
    CHECK(configured_pin == -1 && !v.gate_fault() && !v.frequency_fault() && std::string(v.gate_verdict()) == "off");
  }
#endif

#if TESTBED_SETPOINT_READBACK
  { // A settled reading that matches passes; three that do not in a row trip; one that matches clears.
    reset(); ChannelController c; MeasurementService m; OutputVerifier v(c, m); v.begin();
    c.values[0] = 512; c.changed_ms[0] = 1000;       // 512 of 1023 is 1.65 V
    const float readings[] = {1.66f, 0.0f, 0.0f, 0.0f, 1.65f};
    const bool tripped[] = {false, false, false, true, false};
    for (int i = 0; i < 5; ++i) {
      m.volts[1] = readings[i]; m.at_ms[1] = 2000 + 150 * i;
      v.check(false);
      CHECK(v.readback_fault() == tripped[i]);
    }
  }

  { // A reading taken before the setpoint settled, or before it changed at all, is not judged.
    reset(); ChannelController c; MeasurementService m; OutputVerifier v(c, m); v.begin();
    c.values[1] = 1023; c.changed_ms[1] = 5000;
    for (int i = 0; i < 8; ++i) {
      m.volts[2] = 0.0f; m.at_ms[2] = 4600 + 100 * i;  // 4600 to 5300: before, and too soon after
      v.check(false);
    }
    CHECK(!v.readback_fault() && std::string(v.readback_verdict()) == "ok");
  }

  { // The same reading is only judged once, however many passes see it.
    reset(); ChannelController c; MeasurementService m; OutputVerifier v(c, m); v.begin();
    c.values[2] = 1023; c.changed_ms[2] = 0;
    m.volts[3] = 0.0f; m.at_ms[3] = 9000;
    for (int i = 0; i < 10; ++i) v.check(false);
    CHECK(!v.readback_fault());
  }
#else
  { // Not fitted: never judged.
    reset(); ChannelController c; MeasurementService m; OutputVerifier v(c, m);
    c.values[0] = 1023; m.volts[1] = 0.0f;
    for (int i = 0; i < 5; ++i) { m.at_ms[1] = 2000 + 150 * i; v.check(true); }
    CHECK(!v.readback_fault() && std::string(v.readback_verdict()) == "off");
  }
#endif

  return failures ? 1 : 0;
}
"""


@pytest.mark.skipif(shutil.which("g++") is None, reason="needs a host C++ compiler")
@pytest.mark.parametrize("fitted", [1, 0], ids=["hardware fitted", "not fitted"])
def test_output_verifier_on_the_host(tmp_path, fitted):
    source, binary = tmp_path / "output_verifier.cpp", tmp_path / "output_verifier"
    source.write_text(HARNESS.replace("#include <cstdint>", "#include <cstdint>\n#include <string>").replace("@@FIRMWARE@@", firmware_source()))

    flags = [f"-DTESTBED_GATE_LOOPBACK={fitted}", f"-DTESTBED_SETPOINT_READBACK={fitted}"]
    build = subprocess.run(["g++", "-std=gnu++11", "-Wall", *flags, f"-I{INCLUDE_DIR}", "-o", str(binary), str(source)],
                           capture_output=True, text=True)
    assert build.returncode == 0, build.stderr

    run = subprocess.run([str(binary)], capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stdout + run.stderr
