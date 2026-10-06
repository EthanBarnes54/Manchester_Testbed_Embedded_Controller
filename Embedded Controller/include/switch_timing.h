#pragma once

#include <stdint.h>

// Arithmetic for the hardware-timed switch line. Nothing here touches Arduino or the
// IDF, so the host tests compile this exact file.
//
// An ESP32 LEDC timer counts 0 .. 2^bits - 1 and wraps, advancing once every `divider`
// ticks of its source clock, so one output cycle is divider * 2^bits source ticks. The
// channel holds the line high for the first 2^(bits - 1) counts, which is exactly half.
//
// The divider register also carries 8 fractional bits, but a fractional divider spreads
// its remainder across individual counts and moves edges by up to one source tick. Only
// whole dividers are produced here, so every edge lands where it was asked for, and a
// period with no exact setting is reported as such rather than rounded.

namespace switch_timing {

// ledc_struct.h: the counter is at most 20 bits wide, and clock_divider is 18 bits with
// the low 8 fractional, leaving 1-1023 for a whole-number divider.
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

// Settings for a square wave with period_us between edges (half a cycle, the meaning of
// SWITCH_PERIOD_US) from a source clock of source_hz. exact is false when no whole
// divider and counter width produce that period, and the caller must refuse it.
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

  // Moving every factor of two into the counter leaves the smallest possible divider,
  // so if this one does not fit, no other split will either.
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
