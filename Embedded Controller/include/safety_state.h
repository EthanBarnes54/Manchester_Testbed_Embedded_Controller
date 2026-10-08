#pragma once

#include <stdint.h>

namespace safety {

enum class Mode : uint8_t { Safe, Armed, Fault };

enum class Severity : uint8_t { Warning, Critical };

enum class Fault : uint8_t {
  UnexpectedReset,  // the last reset was a watchdog, a panic or a brownout
  HostTimeout,      // the host went quiet and the failsafe tripped
  ClockConfig,      // APB or REF_TICK is not at the frequency the switch timing assumes
  SwitchGenerator,  // the LEDC channel or the switch timer could not be brought up
  AdcLost,          // the ADS1115 is absent or its conversions are timing out
  LoopOverrun,      // a loop pass took longer than its budget
  LowMemory,        // free heap or the loop task's stack fell below its floor
  SerialOverflow,   // an over-long command line was thrown away
  GateMismatch,     // the gate output disagrees with ARMED and the switch line (loopback)
  SwitchFrequency,  // the gate output's frequency differs from the period set (loopback)
  SetpointMismatch, // a setpoint reads back differently from its command (readback)
  BadChecksum,      // a command arrived whose CRC did not match
  Count
};

constexpr uint8_t FAULT_COUNT = static_cast<uint8_t>(Fault::Count);

inline Severity severity_of(Fault fault) {
  switch (fault) {
    case Fault::UnexpectedReset:
    case Fault::ClockConfig:
    case Fault::SwitchGenerator:
    case Fault::GateMismatch:
    case Fault::SwitchFrequency:
    case Fault::SetpointMismatch:
      return Severity::Critical;
    default:
      return Severity::Warning;
  }
}

// Names as they appear on the wire, in FAULTS lines.
inline const char* name_of(Fault fault) {
  switch (fault) {
    case Fault::UnexpectedReset:
      return "UNEXPECTED_RESET";
    case Fault::HostTimeout:
      return "HOST_TIMEOUT";
    case Fault::ClockConfig:
      return "CLOCK_CONFIG";
    case Fault::SwitchGenerator:
      return "SWITCH_GENERATOR";
    case Fault::AdcLost:
      return "ADC_LOST";
    case Fault::LoopOverrun:
      return "LOOP_OVERRUN";
    case Fault::LowMemory:
      return "LOW_MEMORY";
    case Fault::SerialOverflow:
      return "SERIAL_OVERFLOW";
    case Fault::GateMismatch:
      return "GATE_MISMATCH";
    case Fault::SwitchFrequency:
      return "SWITCH_FREQUENCY";
    case Fault::SetpointMismatch:
      return "SETPOINT_MISMATCH";
    case Fault::BadChecksum:
      return "BAD_CHECKSUM";
    default:
      return "UNKNOWN";
  }
}

inline const char* name_of(Mode mode) {
  switch (mode) {
    case Mode::Armed:
      return "ARMED";
    case Mode::Fault:
      return "FAULT";
    default:
      return "SAFE";
  }
}

inline uint32_t bit_of(Fault fault) {
  return 1UL << static_cast<uint8_t>(fault);
}

class SafetyState {
 public:
  Mode mode() const { return mode_; }
  bool armed() const { return mode_ == Mode::Armed; }

  uint32_t active() const { return active_; }
  uint32_t latched() const { return latched_; }
  uint32_t history() const { return history_; }

  uint16_t count(Fault fault) const { return counts_[static_cast<uint8_t>(fault)]; }

  bool raise(Fault fault) {
    const uint32_t bit = bit_of(fault);
    const uint8_t index = static_cast<uint8_t>(fault);

    if ((active_ & bit) == 0 && counts_[index] < 0xFFFF) {
      ++counts_[index];
    }

    active_ |= bit;
    latched_ |= bit;
    history_ |= bit;

    if (severity_of(fault) != Severity::Critical) {
      return false;
    }

    const bool was_armed = (mode_ == Mode::Armed);
    mode_ = Mode::Fault;
    return was_armed;
  }

  // The condition behind a fault has gone. Its latch stays for the operator to clear.
  void resolve(Fault fault) { active_ &= ~bit_of(fault); }

  const char* arm_refusal() const {
    if (critical_in(active_)) {
      return "a critical fault is active";
    }

    if (mode_ == Mode::Fault || critical_in(latched_)) {
      return "a critical fault is latched, send CLEAR FAULTS";
    }

    return nullptr;
  }

  bool arm() {
    if (arm_refusal() != nullptr) {
      return false;
    }

    mode_ = Mode::Armed;
    return true;
  }

  bool disarm() {
    if (mode_ != Mode::Armed) {
      return false;
    }

    mode_ = Mode::Safe;
    return true;
  }

  void clear_latched() {
    latched_ &= active_;

    if (mode_ == Mode::Fault && !critical_in(latched_)) {
      mode_ = Mode::Safe;
    }
  }

  void clear_history() {
    history_ = latched_;
  }

  // Restores the history saved from an earlier boot.
  void restore_history(uint32_t saved) {
    history_ |= saved & all_faults();
  }

 private:
  Mode mode_ = Mode::Safe;
  uint32_t active_ = 0;
  uint32_t latched_ = 0;
  uint32_t history_ = 0;
  uint16_t counts_[FAULT_COUNT] = {0};

  static uint32_t all_faults() { return (1UL << FAULT_COUNT) - 1; }

  static bool critical_in(uint32_t faults) {
    for (uint8_t index = 0; index < FAULT_COUNT; ++index) {
      if ((faults & (1UL << index)) && severity_of(static_cast<Fault>(index)) == Severity::Critical) {
        return true;
      }
    }

    return false;
  }
};

}  // namespace safety
