# 0008 Compile output verification in, but switch it on per build

Status: accepted, 4.6.6.

## Context

Commanding an output is not the same as the output doing it: a gate can fail, a wire can
come off, a setpoint stage can saturate. Two checks close the loop: the gate output looped
back to GPIO 34 and counted by the pulse counter, and setpoints 1-3 read back on the
ADS1115's spare inputs. Both need hardware that a bare development board does not have,
and a check without its hardware would raise false critical faults.

## Decision

Both checks are always compiled. Each stays off (`off` in `SELFTEST` and `HEALTH`) unless its
build flag, `TESTBED_GATE_LOOPBACK` or `TESTBED_SETPOINT_READBACK`, is 1. The deployment
environment, `esp32_deploy`, sets both. `VERSION` reports which the build expects.

## Reasons

- One source for every build: the checks are compiled and host-tested whether or not a
  given bench has the hardware, so they cannot rot unbuilt.
- The flags are `constexpr` branches rather than `#if`, so the compiler checks both sides.
- The build that goes on the rig cannot be one that skips the checks by accident.

## Consequences

- The deployment build cannot run on a bench without the loopback and readback wired; use
  `esp32dev` there.
- The readback scale, tolerance and settling time are constants that must match the
  board's network before the check means anything (README).
