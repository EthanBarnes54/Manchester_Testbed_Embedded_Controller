#pragma once

#include <stdint.h>

// Loop-pass timing for the HEALTH report and the loop-overrun fault. Pure logic with no
// Arduino dependency, so the host tests compile this exact file.
//
// The window maximum is what HEALTH reports and then resets, so each report shows the
// worst pass since the previous one. The peak is the worst since boot, and overruns
// counts passes that took longer than the budget.

namespace health {

class LoopTiming {
 public:
  explicit LoopTiming(uint32_t budget_us) : budget_us_(budget_us) {}

  void record(uint32_t pass_us) {
    if (pass_us > window_max_us_) {
      window_max_us_ = pass_us;
    }

    if (pass_us > peak_us_) {
      peak_us_ = pass_us;
    }

    if (pass_us > budget_us_ && overruns_ < 0xFFFFFFFFUL) {
      ++overruns_;
    }

    if (passes_ < 0xFFFFFFFFUL) {
      ++passes_;
    }
  }

  // The worst pass since the last call, then starts a new window.
  uint32_t take_window_max() {
    const uint32_t worst = window_max_us_;
    window_max_us_ = 0;
    return worst;
  }

  uint32_t peak_us() const { return peak_us_; }
  uint32_t overruns() const { return overruns_; }
  uint32_t passes() const { return passes_; }
  uint32_t budget_us() const { return budget_us_; }

 private:
  uint32_t budget_us_;
  uint32_t window_max_us_ = 0;
  uint32_t peak_us_ = 0;
  uint32_t overruns_ = 0;
  uint32_t passes_ = 0;
};

}  // namespace health
