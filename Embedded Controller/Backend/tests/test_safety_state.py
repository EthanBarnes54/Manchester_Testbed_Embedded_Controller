"""The firmware's safety state machine, compiled from include/safety_state.h and run on the host."""

import shutil
import subprocess
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
INCLUDE_DIR = BACKEND_DIR.parent / "include"

HARNESS = r"""
#include <cstdio>
#include <cstring>
#include "safety_state.h"

using namespace safety;

static int failures = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++failures; } } while (0)

int main() {
  { // Boots Safe; ARM and DISARM move between Safe and Armed and say whether outputs must drop.
    SafetyState s;
    CHECK(s.mode() == Mode::Safe && !s.armed());
    CHECK(s.arm_refusal() == nullptr);
    CHECK(s.arm() && s.armed());
    CHECK(s.arm() && s.armed());                       // ARM while armed is accepted
    CHECK(s.disarm() && s.mode() == Mode::Safe);       // was armed: outputs must drop
    CHECK(!s.disarm() && s.mode() == Mode::Safe);      // already safe
  }

  { // A warning is recorded but never disarms or blocks ARM.
    SafetyState s;
    s.arm();
    CHECK(!s.raise(Fault::HostTimeout));
    CHECK(s.armed());
    CHECK(s.active() == bit_of(Fault::HostTimeout) && s.latched() == bit_of(Fault::HostTimeout));
    s.disarm();
    CHECK(s.arm_refusal() == nullptr && s.arm());
  }

  { // A critical fault while armed moves to Fault and tells the caller to drop the outputs.
    SafetyState s;
    s.arm();
    CHECK(s.raise(Fault::UnexpectedReset));
    CHECK(s.mode() == Mode::Fault && !s.armed());
    CHECK(!s.disarm() && s.mode() == Mode::Fault);     // DISARM never leaves Fault

    // Refused while the condition is present, refused while only latched, then accepted.
    CHECK(!s.arm() && std::strstr(s.arm_refusal(), "active") != nullptr);
    s.clear_latched();
    CHECK(s.mode() == Mode::Fault);                    // still active, so the latch stays
    s.resolve(Fault::UnexpectedReset);
    CHECK(!s.arm() && std::strstr(s.arm_refusal(), "CLEAR FAULTS") != nullptr);
    s.clear_latched();
    CHECK(s.mode() == Mode::Safe && s.latched() == 0);
    CHECK(s.arm());
  }

  { // A critical fault while safe still latches Fault, and raise() says nothing was armed.
    SafetyState s;
    CHECK(!s.raise(Fault::UnexpectedReset));
    CHECK(s.mode() == Mode::Fault);
  }

  { // Counted once per onset, not per repeat while active, and saturating.
    SafetyState s;
    s.raise(Fault::HostTimeout);
    s.raise(Fault::HostTimeout);
    CHECK(s.count(Fault::HostTimeout) == 1);
    s.resolve(Fault::HostTimeout);
    s.raise(Fault::HostTimeout);
    CHECK(s.count(Fault::HostTimeout) == 2);
    for (int i = 0; i < 70000; ++i) { s.resolve(Fault::HostTimeout); s.raise(Fault::HostTimeout); }
    CHECK(s.count(Fault::HostTimeout) == 0xFFFF);
  }

  { // History outlives CLEAR FAULTS; CLEAR LOG keeps only what is still latched; restore ignores unknown bits.
    SafetyState s;
    s.raise(Fault::HostTimeout);
    s.resolve(Fault::HostTimeout);
    s.clear_latched();
    CHECK(s.latched() == 0 && s.history() == bit_of(Fault::HostTimeout));
    s.clear_history();
    CHECK(s.history() == 0);

    SafetyState t;
    t.restore_history(0xFFFFFFFFUL);
    CHECK(t.history() == ((1UL << FAULT_COUNT) - 1));
    CHECK(t.latched() == 0 && t.mode() == Mode::Safe);  // history alone never latches
  }

  { // Only a fault that means the outputs cannot be trusted, or a reset nobody asked for, is critical.
    const Fault critical[] = {Fault::UnexpectedReset, Fault::ClockConfig, Fault::SwitchGenerator,
                             Fault::GateMismatch, Fault::SwitchFrequency, Fault::SetpointMismatch};
    for (uint8_t i = 0; i < FAULT_COUNT; ++i) {
      bool expected = false;
      for (Fault f : critical) expected = expected || static_cast<uint8_t>(f) == i;
      CHECK((severity_of(static_cast<Fault>(i)) == Severity::Critical) == expected);
    }
  }

  { // Every fault has a wire name.
    for (uint8_t i = 0; i < FAULT_COUNT; ++i) CHECK(std::strcmp(name_of(static_cast<Fault>(i)), "UNKNOWN") != 0);
    CHECK(std::strcmp(name_of(Mode::Fault), "FAULT") == 0);
  }

  return failures ? 1 : 0;
}
"""


@pytest.mark.skipif(shutil.which("g++") is None, reason="needs a host C++ compiler")
def test_safety_state_on_the_host(tmp_path):
    source = tmp_path / "safety_state.cpp"
    binary = tmp_path / "safety_state"
    source.write_text(HARNESS)

    build = subprocess.run(
        ["g++", "-std=gnu++11", "-Wall", "-Wextra", "-Werror", f"-I{INCLUDE_DIR}", "-o", str(binary), str(source)],
        capture_output=True,
        text=True,
    )
    assert build.returncode == 0, build.stderr

    run = subprocess.run([str(binary)], capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stdout + run.stderr
