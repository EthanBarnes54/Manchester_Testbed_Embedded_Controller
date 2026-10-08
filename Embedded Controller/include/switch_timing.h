#pragma once

#include <stdint.h>

namespace switch_timing {

constexpr uint32_t LEDC_MAX_COUNTER_BITS = 20;
constexpr uint32_t LEDC_MAX_DIVIDER = 1023;
constexpr uint32_t LEDC_DIVIDER_FRACTIONAL_BITS = 8;

struct LedcWaveform {
  bool exact;
  uint32_t counter_bits;  // one cycle is 2^counter_bits counts
  uint32_t divider;       // whole source-clock ticks per count
  uint32_t high_counts;   // counts spent high, exactly half the cycle

  // The value ledc_timer_set() writes straight into the 10.8 fixed-point register.
  uint32_t divider_register() const {
    return divider << LEDC_DIVIDER_FRACTIONAL_BITS;
  }
};

inline LedcWaveform ledc_waveform_for(uint32_t period_us, uint32_t source_hz) {
  const LedcWaveform rejected = {false, 0, 0, 0};

  if (period_us == 0 || source_hz == 0) {
    return rejected;
  }

  const uint64_t cycle_ticks_scaled = 2ULL * period_us * source_hz;

  if (cycle_ticks_scaled % 1000000ULL != 0) {
    return rejected;
  }

  uint64_t divider = cycle_ticks_scaled / 1000000ULL;
  uint32_t bits = 0;

  while (bits < LEDC_MAX_COUNTER_BITS && (divider & 1ULL) == 0) {
    divider >>= 1;
    ++bits;
  }

  if (bits == 0 || divider > LEDC_MAX_DIVIDER) {
    return rejected;
  }

  const LedcWaveform waveform = {true, bits, static_cast<uint32_t>(divider), 1UL << (bits - 1)};
  return waveform;
}

}  // namespace switch_timing
