"""The firmware's loop-timing statistics, compiled from include/health_stats.h and run on the host."""

import pytest

from host_build import build_and_run, needs_compiler


HARNESS = r"""
#include <cstdio>
#include "health_stats.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++failures; } } while (0)

int main() {
  health::LoopTiming t(20000);
  CHECK(t.take_window_max() == 0 && t.peak_us() == 0 && t.overruns() == 0 && t.budget_us() == 20000);

  t.record(1100); t.record(4200); t.record(900);
  CHECK(t.take_window_max() == 4200);                 // worst since the last report
  CHECK(t.take_window_max() == 0);                    // and the window starts again

  t.record(1500);
  CHECK(t.take_window_max() == 1500 && t.peak_us() == 4200);  // the peak is since boot
  CHECK(t.overruns() == 0);

  t.record(20000);                                    // at the budget is not over it
  t.record(20001); t.record(60000);
  CHECK(t.overruns() == 2 && t.peak_us() == 60000 && t.passes() == 7);

  return failures ? 1 : 0;
}
"""


@pytest.mark.req("BIT-02")
@needs_compiler
def test_loop_timing_on_the_host(tmp_path):
    run = build_and_run(HARNESS, tmp_path, "health")
    assert run.returncode == 0, run.stdout + run.stderr
