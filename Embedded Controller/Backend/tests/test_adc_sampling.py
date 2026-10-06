"""The firmware's ADC sampling, compiled from main.cpp and run on the host against a fake ADS1115.

The fake takes 8 ms per conversion like the real part at 128 SPS, can vanish from the bus,
and flags any conversion restarted while one is still running.
"""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
FIRMWARE_DIR = BACKEND_DIR.parent
MAIN_CPP = (FIRMWARE_DIR / "src" / "main.cpp").read_text(encoding="utf-8")


def firmware_source():
    names = ("MEASUREMENT_INTERVAL_MS", "ADC_FIRST_POLL_MS", "ADC_CONVERSION_TIMEOUT_MS", "MEASURED_DECIMAL_PLACES")
    lines = [re.search(rf"^constexpr [\w ]+ {name} = [^;]+;", MAIN_CPP, re.M).group(0) for name in names]
    lines.append(re.search(r"^constexpr adsGain_t ADC_GAIN = [^;]+;", MAIN_CPP, re.M).group(0))
    lines.append(re.search(r"^#ifndef TESTBED_SETPOINT_READBACK\n.*?^#endif", MAIN_CPP, re.S | re.M).group(0))
    for name in ("SETPOINT_READBACK_FITTED", "READBACK_INPUTS", "READBACK_START_WINDOW_MS"):
        lines.append(re.search(rf"^constexpr [\w ]+ {name} = [^;]+;", MAIN_CPP, re.M).group(0))
    lines.append(re.search(r"^class MeasurementService \{.*?^\};", MAIN_CPP, re.S | re.M).group(0))
    return "\n".join(lines)


HARNESS = r"""
#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>

// ---- Stand-ins for Arduino's String and Serial ----

struct String {
  std::string s;
  String(const char* text) : s(text) {}
  String(float value, unsigned places) { char b[32]; std::snprintf(b, sizeof b, "%.*f", (int)places, value); s = b; }
  String(const std::string& text) : s(text) {}
};
String operator+(const String& a, const String& b) { return String(a.s + b.s); }
String operator+(const String& a, const char* b) { return String(a.s + b); }
String operator+(const String& a, unsigned int b) { return String(a.s + std::to_string(b)); }
String operator+(const String& a, unsigned long b) { return String(a.s + std::to_string(b)); }

static std::vector<std::string> lines;
struct SerialPort {
  void println(const String& line) { lines.push_back(line.s); }
  void println(const char* line) { lines.push_back(line); }
} Serial;

// ---- A fake ADS1115 with the library's interface ----

typedef int adsGain_t;
#define GAIN_ONE 0x0200
static const uint16_t MUX_BY_CHANNEL[] = {0x4000, 0x5000, 0x6000, 0x7000};

static unsigned long now_ms = 0;
static int calls_this_update = 0;

struct Adafruit_ADS1115 {
  bool present = true;
  bool converting = false;
  bool restarted_mid_conversion = false;
  unsigned long conversion_ms = 8;
  unsigned long started_ms = 0;
  int starts = 0, polls = 0;
  uint16_t last_mux = 0;
  int starts_by_input[4] = {0};
  unsigned long last_diode_start = 0, worst_readback_lag = 0;

  void setGain(adsGain_t) {}
  bool begin() { return present; }
  void startADCReading(uint16_t mux, bool continuous) {
    ++calls_this_update;
    if (!present) return;
    if (converting) restarted_mid_conversion = true;
    converting = !continuous; started_ms = now_ms; ++starts; last_mux = mux;
    const int input = (mux - 0x4000) >> 12;
    ++starts_by_input[input];
    if (input == 0) last_diode_start = now_ms;
    else if (now_ms - last_diode_start > worst_readback_lag) worst_readback_lag = now_ms - last_diode_start;
  }
  bool conversionComplete() {
    ++calls_this_update; ++polls;
    if (!present || !converting) return false;
    if (now_ms - started_ms >= conversion_ms) { converting = false; return true; }
    return false;
  }
  // The diode reads 10000 counts (1.25 V); readback input n reads 1000 x n.
  int16_t getLastConversionResults() { ++calls_this_update; const int input = (last_mux - 0x4000) >> 12; return input == 0 ? 10000 : 1000 * input; }
  float computeVolts(int16_t counts) { return counts * 4.096f / 32768.0f; }
};

unsigned long millis() { return now_ms; }
void send_line(const String& line) { lines.push_back(line.s); }

// ---- The firmware under test ----

@@FIRMWARE@@

// ---- Scenarios ----

static int failures = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d at %lu ms: %s\n", __LINE__, now_ms, #c); ++failures; } } while (0)

static int count(const std::string& prefix) {
  int n = 0;
  for (const std::string& l : lines) if (l.compare(0, prefix.size(), prefix) == 0) ++n;
  return n;
}

static int most_calls = 0;
static void run_to(MeasurementService& m, unsigned long t) {
  while (now_ms < t) {
    ++now_ms;
    calls_this_update = 0;
    m.update(now_ms);
    if (calls_this_update > most_calls) most_calls = calls_this_update;
  }
}

int main() {
  { // A steady 20 Hz stream, one formatter, never more than two bus calls in a pass.
    Adafruit_ADS1115 adc; MeasurementService m(adc); lines.clear(); now_ms = 0; most_calls = 0;
    run_to(m, 1000);
    CHECK(count("MEASURED ") >= 19 && count("MEASURED ") <= 20);
    CHECK(lines.size() == static_cast<size_t>(count("MEASURED ")));
    // Numbered from 1 and stamped with the board's clock, so the host can see gaps and restarts.
    CHECK(lines[0] == "MEASURED 1.25000 V seq=1 t_ms=" + std::to_string(58));
    CHECK(lines[1].compare(0, 25, "MEASURED 1.25000 V seq=2 ") == 0);
    // Every diode conversion is reported, bar one that may still be in flight at the end.
    CHECK(!adc.restarted_mid_conversion && adc.starts_by_input[0] - count("MEASURED ") == (adc.converting ? 1 : 0));
    CHECK(adc.polls <= 2 * adc.starts);  // nothing is polled before the conversion can be ready
    CHECK(most_calls <= 2);
    CHECK(!m.lost() && m.conversions() + (adc.converting ? 1 : 0) == static_cast<uint32_t>(adc.starts) && m.timeouts() == 0);
  }

  { // READ during a conversion is answered by that conversion and does not restart it.
    Adafruit_ADS1115 adc; MeasurementService m(adc); lines.clear(); now_ms = 0;
    run_to(m, 52);                       // a conversion started at 50 ms is in flight
    CHECK(adc.converting && adc.starts_by_input[0] == 1);
    m.request_reading();
    run_to(m, 60);
    CHECK(adc.starts_by_input[0] == 1 && !adc.restarted_mid_conversion);
    CHECK(count("MEASURED ") == 1);
    run_to(m, 99);                       // and the stream carries on from its own schedule
    CHECK(adc.starts_by_input[0] == 1);
    run_to(m, 100);
    CHECK(adc.starts_by_input[0] == 2);
  }

  { // READ with the ADC idle starts a conversion at once instead of waiting for the stream.
    Adafruit_ADS1115 adc; MeasurementService m(adc); lines.clear(); now_ms = 0;
    run_to(m, 70);
    CHECK(!adc.converting && count("MEASURED ") == 1);
    m.request_reading();
    run_to(m, 71);
    CHECK(adc.converting && adc.starts_by_input[0] == 2);
    run_to(m, 80);
    CHECK(count("MEASURED ") == 2 && !adc.restarted_mid_conversion);
  }

  { // An ADC that is gone never stops the loop: one error per outage, and READ always gets an answer.
    Adafruit_ADS1115 adc; adc.present = false; MeasurementService m(adc); lines.clear(); now_ms = 0; most_calls = 0;
    run_to(m, 1000);
    CHECK(now_ms == 1000 && most_calls <= 2);
    CHECK(count("MEASURED ") == 0);
    CHECK(count("ERROR: ADC conversion timed out!") == 1);
    CHECK(m.lost() && m.timeouts() >= 19 && m.conversions() == 0);  // every attempt counted, reported once
    m.request_reading();
    run_to(m, 1100);
    CHECK(count("ERROR: ADC conversion timed out!") == 2);

    adc.present = true;                  // it comes back
    run_to(m, 1300);
    CHECK(count("MEASURED ") >= 3);
    CHECK(!m.lost());                    // noticed without a reset
    adc.present = false;                 // and goes again: reported afresh
    run_to(m, 1500);
    CHECK(count("ERROR: ADC conversion timed out!") == 3);
  }

#if TESTBED_SETPOINT_READBACK
  { // Readback shares the ADC without touching the diode stream: each input is read in turn,
    // only ever straight after a diode conversion, and lands in its own slot.
    Adafruit_ADS1115 adc; MeasurementService m(adc); lines.clear(); now_ms = 0;
    run_to(m, 1000);
    CHECK(count("MEASURED ") >= 19 && count("MEASURED ") <= 20);
    CHECK(!adc.restarted_mid_conversion);
    for (int input = 1; input <= 3; ++input) {
      CHECK(adc.starts_by_input[input] >= 5);
      CHECK(m.readback_ms(input) > 0);
      CHECK(m.readback_volts(input) == adc.computeVolts(static_cast<int16_t>(1000 * input)));
    }
    CHECK(adc.worst_readback_lag <= READBACK_START_WINDOW_MS);
    CHECK(lines.size() == static_cast<size_t>(count("MEASURED ")));   // readbacks are never printed
  }

  { // An ADC slow enough that a readback could not finish before the next diode conversion
    // gets none: the diode stream comes first.
    Adafruit_ADS1115 adc; adc.conversion_ms = READBACK_START_WINDOW_MS + 2; MeasurementService m(adc);
    lines.clear(); now_ms = 0;
    run_to(m, 500);
    CHECK(adc.worst_readback_lag <= READBACK_START_WINDOW_MS);
    CHECK(adc.starts == adc.starts_by_input[0] && count("MEASURED ") >= 9);
  }

  { // A READ that arrives during a readback conversion is answered by the next diode conversion.
    Adafruit_ADS1115 adc; MeasurementService m(adc); lines.clear(); now_ms = 0;
    run_to(m, 60);                       // diode at 50 done by 58, readback started straight after
    CHECK(adc.converting && (adc.last_mux - 0x4000) >> 12 != 0);
    m.request_reading();
    run_to(m, 80);
    CHECK(count("MEASURED ") == 2 && !adc.restarted_mid_conversion);
  }
#else
  { // Not fitted: only the diode is ever converted.
    Adafruit_ADS1115 adc; MeasurementService m(adc); lines.clear(); now_ms = 0;
    run_to(m, 500);
    CHECK(adc.starts == adc.starts_by_input[0] && m.readback_ms(1) == 0);
  }
#endif

  return failures ? 1 : 0;
}
"""


@pytest.mark.skipif(shutil.which("g++") is None, reason="needs a host C++ compiler")
@pytest.mark.parametrize("readback", [0, 1], ids=["readback not fitted", "readback fitted"])
def test_adc_sampling_on_the_host(tmp_path, readback):
    source = tmp_path / "adc_sampling.cpp"
    binary = tmp_path / "adc_sampling"
    source.write_text(HARNESS.replace("@@FIRMWARE@@", firmware_source()))

    build = subprocess.run(["g++", "-std=gnu++11", "-Wall", f"-DTESTBED_SETPOINT_READBACK={readback}", "-o", str(binary), str(source)],
                           capture_output=True, text=True)
    assert build.returncode == 0, build.stderr

    run = subprocess.run([str(binary)], capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stdout + run.stderr


def test_nothing_in_the_firmware_waits_on_a_conversion():
    assert ".readADC_SingleEnded(" not in MAIN_CPP
    assert "measurement_.request_reading();" in MAIN_CPP
