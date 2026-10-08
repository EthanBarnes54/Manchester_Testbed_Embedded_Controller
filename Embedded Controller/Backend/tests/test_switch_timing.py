"""The switch line's LEDC arithmetic, compiled from include/switch_timing.h and run on the host.

The harness checks every whole period against an independent brute-force search, so a
period is only ever reported exact when the hardware would really produce it, and only
ever rejected when no whole divider exists.
"""

import re
import shutil

import pytest

from host_build import COMPILER, FIRMWARE_DIR, build, run

APB_HZ = 80_000_000
REF_TICK_HZ = 1_000_000
LARGEST_PERIOD_US = 2_000_000

HARNESS = r"""
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "switch_timing.h"

using switch_timing::LedcWaveform;
using switch_timing::ledc_waveform_for;

// Independent of the header: try every counter width the hardware has.
static bool whole_divider_exists(unsigned long long cycle_ticks) {
  for (unsigned bits = 1; bits <= 20; ++bits) {
    const unsigned long long counts = 1ULL << bits;
    if (cycle_ticks % counts == 0 && cycle_ticks / counts >= 1 && cycle_ticks / counts <= 1023) return true;
  }
  return false;
}

int main(int argc, char** argv) {
  if (argc == 4 && std::strcmp(argv[1], "one") == 0) {
    const LedcWaveform w = ledc_waveform_for(std::strtoul(argv[2], 0, 10), std::strtoul(argv[3], 0, 10));
    std::printf("%d %u %u %u %u\n", w.exact ? 1 : 0, w.counter_bits, w.divider, w.high_counts, w.divider_register());
    return 0;
  }

  if (argc != 4 || std::strcmp(argv[1], "sweep") != 0) return 2;

  const unsigned long source_hz = std::strtoul(argv[2], 0, 10);
  const unsigned long last_period = std::strtoul(argv[3], 0, 10);
  unsigned long first_rejected = 0, exact_count = 0, failures = 0;

  for (unsigned long period = 1; period <= last_period; ++period) {
    const LedcWaveform w = ledc_waveform_for(period, source_hz);
    const unsigned long long scaled = 2ULL * period * source_hz;
    const bool representable = scaled % 1000000ULL == 0 && whole_divider_exists(scaled / 1000000ULL);

    if (w.exact != representable) {
      if (failures++ < 5) std::printf("MISMATCH period %lu: exact=%d representable=%d\n", period, w.exact, representable);
      continue;
    }

    if (!w.exact) {
      if (first_rejected == 0) first_rejected = period;
      continue;
    }

    ++exact_count;
    const unsigned long long counts = 1ULL << w.counter_bits;
    const bool in_range = w.counter_bits >= 1 && w.counter_bits <= 20 && w.divider >= 1 && w.divider <= 1023;
    const bool half_duty = 2ULL * w.high_counts == counts;
    // The high phase alone must last exactly period_us: high_counts * divider source ticks.
    const bool high_phase_exact = 1000000ULL * w.high_counts * w.divider == 1ULL * period * source_hz;
    const bool register_whole = w.divider_register() == (w.divider << 8) && (w.divider_register() & 0xFF) == 0;

    if (!(in_range && half_duty && high_phase_exact && register_whole)) {
      if (failures++ < 5) std::printf("BAD period %lu: bits %u divider %u high %u\n", period, w.counter_bits, w.divider, w.high_counts);
    }
  }

  std::printf("first_rejected=%lu exact=%lu failures=%lu\n", first_rejected, exact_count, failures);
  return failures ? 1 : 0;
}
"""


def make_harness(work, include_dirs=(), **options):
    """Builds the LEDC arithmetic harness; returns a function that runs it and returns (exit code, output)."""

    # -O2 because the sweep covers every period up to 2 s.
    binary = build(HARNESS, work, "switch_timing_harness", include_dirs=include_dirs, optimise=True, **options)

    def invoke(*args):
        result = run(binary, *args)
        return result.returncode, result.stdout.strip()

    return invoke


@pytest.fixture(scope="module")
def harness(tmp_path_factory):
    if shutil.which(COMPILER) is None:
        pytest.skip("needs a host C++ compiler")

    return make_harness(tmp_path_factory.mktemp("switch_timing"))


def waveform(harness, period_us, source_hz):
    code, out = harness("one", period_us, source_hz)
    assert code == 0, out
    exact, bits, divider, high, register = map(int, out.split())
    return {"exact": bool(exact), "bits": bits, "divider": divider, "high": high, "register": register}


def sweep(harness, source_hz):
    code, out = harness("sweep", source_hz, LARGEST_PERIOD_US)
    assert code == 0, out
    return dict(item.split("=") for item in out.splitlines()[-1].split())


@pytest.mark.req("SW-02")
@pytest.mark.parametrize("source_hz", [REF_TICK_HZ, APB_HZ])
def test_every_period_is_exact_or_rejected_never_rounded(harness, source_hz):
    assert sweep(harness, source_hz)["failures"] == "0"


@pytest.mark.req("SW-01")
def test_ref_tick_covers_every_period_up_to_1024_us(harness):
    # Half a cycle on the 1 MHz REF_TICK is period_us ticks, so the divider is just the
    # odd part of the period, and every odd part up to 1023 fits.
    assert sweep(harness, REF_TICK_HZ)["first_rejected"] == "1025"


@pytest.mark.req("SW-01")
def test_apb_runs_out_of_divider_at_205_us(harness):
    # 80 MHz carries a factor of 5 that has to go into the divider, so odd parts above
    # 204 overflow 1023. This is why the switch is clocked from REF_TICK.
    assert sweep(harness, APB_HZ)["first_rejected"] == "205"


@pytest.mark.req("SW-02")
def test_every_period_the_firmware_sends_to_the_ledc_is_exact(harness):
    main_cpp = (FIRMWARE_DIR / "src" / "main.cpp").read_text(encoding="utf-8")

    def constant(name):
        return re.search(rf"constexpr\s+[\w:\s]+?\s{name}\s*=\s*([^;]+);", main_cpp).group(1).strip()

    # The firmware clocks the switch from REF_TICK, which soc.h defines as 1 MHz.
    assert constant("SWITCH_LEDC_CLOCK_HZ") == "REF_CLK_FREQ"
    assert "ledc_waveform_for(period_us, SWITCH_LEDC_CLOCK_HZ)" in main_cpp
    assert "waveform.counter_bits, LEDC_REF_TICK)" in main_cpp

    floor_us, ceiling_us = int(constant("SWITCH_PERIOD_MIN_US")), int(constant("SWITCH_HARDWARE_MAX_US"))
    assert 1 <= floor_us <= ceiling_us
    assert ceiling_us < int(sweep(harness, REF_TICK_HZ)["first_rejected"])


@pytest.mark.req("SW-01")
@pytest.mark.parametrize(
    "period_us, source_hz, bits, divider",
    [
        (5, APB_HZ, 5, 25),  # 800 ticks a cycle = 25 x 2^5, duty 16/32
        (1, APB_HZ, 5, 5),  # 160 ticks a cycle
        (5, REF_TICK_HZ, 1, 5),
        (1, REF_TICK_HZ, 1, 1),
        (1000, REF_TICK_HZ, 4, 125),
        (1024, REF_TICK_HZ, 11, 1),
    ],
)
def test_known_settings(harness, period_us, source_hz, bits, divider):
    w = waveform(harness, period_us, source_hz)

    assert (w["exact"], w["bits"], w["divider"]) == (True, bits, divider)
    assert w["high"] == 2 ** (bits - 1)
    assert w["register"] == divider << 8


# The last two use a source that does not tick in whole microseconds (a 32.768 kHz
# crystal), where a period truncated to whole ticks would otherwise be passed off as exact.
@pytest.mark.req("SW-02")
@pytest.mark.parametrize("period_us, source_hz", [(0, REF_TICK_HZ), (1025, REF_TICK_HZ), (205, APB_HZ), (5, 0),
                                                  (1, 32_768), (1000, 32_768)])
def test_unrepresentable_periods_are_rejected(harness, period_us, source_hz):
    assert waveform(harness, period_us, source_hz)["exact"] is False
