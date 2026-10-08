#pragma once

#include <math.h>
#include <stdint.h>

namespace output_checks {

enum class GateExpectation : uint8_t { Low, High, Switching };

inline GateExpectation expected_gate(bool armed, bool switching, bool held_high) {
  if (!armed) {
    return GateExpectation::Low;
  }

  if (switching) {
    return GateExpectation::Switching;
  }

  return held_high ? GateExpectation::High : GateExpectation::Low;
}

// One rising edge per cycle, and a cycle is two periods (period_us is between edges).
inline uint32_t expected_rising_edges(uint64_t elapsed_us, uint32_t period_us) {
  return period_us == 0 ? 0 : static_cast<uint32_t>(elapsed_us / (2ULL * period_us));
}

inline uint32_t frequency_window_us(uint32_t period_us) {
  const uint64_t four_cycles = 8ULL * period_us;
  return four_cycles > 20000ULL ? static_cast<uint32_t>(four_cycles) : 20000UL;
}

inline bool edge_count_plausible(uint32_t counted, uint32_t expected) {
  const uint32_t tolerance = expected / 200 > 2 ? expected / 200 : 2;
  const uint32_t difference = counted > expected ? counted - expected : expected - counted;
  return difference <= tolerance;
}

inline bool readback_matches(float expected_v, float measured_v, float tolerance_v) {
  return fabsf(expected_v - measured_v) <= tolerance_v;
}

class Debounce {
 public:
  explicit Debounce(uint8_t needed) : needed_(needed) {}

  bool update(bool disagrees) {
    if (!disagrees) {
      run_ = 0;
    } else if (run_ < 0xFF) {
      ++run_;
    }

    return run_ >= needed_;
  }

  bool tripped() const { return run_ >= needed_; }

 private:
  uint8_t needed_;
  uint8_t run_ = 0;
};

}  // namespace output_checks
