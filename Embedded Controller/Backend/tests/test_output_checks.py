import pytest

from host_build import build_and_run, needs_compiler


HARNESS = r"""
#include <cstdio>
#include "output_checks.h"

using namespace output_checks;

static int failures = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++failures; } } while (0)

int main() {
  // Disarmed the gate is closed whatever the line does; armed it passes the line.
  CHECK(expected_gate(false, true, true) == GateExpectation::Low);
  CHECK(expected_gate(false, false, true) == GateExpectation::Low);
  CHECK(expected_gate(true, false, true) == GateExpectation::High);
  CHECK(expected_gate(true, false, false) == GateExpectation::Low);
  CHECK(expected_gate(true, true, false) == GateExpectation::Switching);

  // One rising edge per cycle, and a cycle is two periods.
  CHECK(expected_rising_edges(20000, 1) == 10000);
  CHECK(expected_rising_edges(20000, 5) == 2000);
  CHECK(expected_rising_edges(16000000ULL, 2000000) == 4);
  CHECK(expected_rising_edges(20000, 0) == 0);

  // About four cycles per window, never under 20 ms, and the count always fits 16 bits.
  CHECK(frequency_window_us(1) == 20000);
  CHECK(frequency_window_us(2500) == 20000);
  CHECK(frequency_window_us(2501) == 20008);
  CHECK(frequency_window_us(2000000) == 16000000);
  for (uint32_t period = 1; period <= 2000000; period = period < 1000 ? period + 1 : period * 2) {
    const uint32_t window = frequency_window_us(period);
    CHECK(expected_rising_edges(window, period) <= 32767);
    CHECK(expected_rising_edges(window, period) >= 4);
  }

  // Two edges either way is noise; beyond that, 0.5% of a large count.
  CHECK(edge_count_plausible(4, 4) && edge_count_plausible(2, 4) && edge_count_plausible(6, 4));
  CHECK(!edge_count_plausible(1, 4) && !edge_count_plausible(7, 4));
  CHECK(edge_count_plausible(10050, 10000) && edge_count_plausible(9950, 10000));
  CHECK(!edge_count_plausible(10051, 10000) && !edge_count_plausible(9949, 10000));
  CHECK(!edge_count_plausible(5000, 10000));                          // half the frequency
  CHECK(!edge_count_plausible(0, 10000));                             // nothing coming through

  CHECK(readback_matches(1.65f, 1.70f, 0.15f) && !readback_matches(1.65f, 1.85f, 0.15f));

  // Only a run of disagreements trips, and one agreement resets the run.
  Debounce d(3);
  CHECK(!d.update(true) && !d.update(true));
  CHECK(!d.update(false) && !d.update(true) && !d.update(true));
  CHECK(d.update(true) && d.tripped());
  CHECK(!d.update(false) && !d.tripped());

  return failures ? 1 : 0;
}
"""


@pytest.mark.req("OUT-01", "OUT-02")
@needs_compiler
def test_output_checks_on_the_host(tmp_path):
    run = build_and_run(HARNESS, tmp_path, "output_checks")
    assert run.returncode == 0, run.stdout + run.stderr
