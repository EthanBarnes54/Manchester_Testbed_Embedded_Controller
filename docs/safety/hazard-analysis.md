# Hazard analysis

What can go wrong with the controller that matters outside it, how each hazard is
mitigated, and which requirements and checks show the mitigation works. It covers the
controller (board, firmware, backend, dashboard), not the rig it drives.

**Severities are provisional.** What the five setpoints and the switch line ultimately
drive (supplies, a high-voltage switch, the beam) is decided by the rig, not here, and the
gate's load is still undecided (README). The rig's safety owner should set each severity
and accept each residual risk. The scale follows MIL-STD-882: Catastrophic, Critical,
Marginal, Negligible.

`tests/test_docs.py` checks that every requirement ID named here exists.

## Hazards

| ID | Hazard | Causes | Mitigations | Requirements | Severity (provisional) | Residual risk |
|---|---|---|---|---|---|---|
| H-01 | An output goes live at power-on or reset | GPIO 16 is driven by the PSRAM probe before any firmware runs; outputs undriven during boot | AND gate enabled only by ARMED, pulled down (ADR 0004). ARMED and the switch line driven low before anything else. The board starts SAFE | SW-03, SAF-01 | Critical | Depends on the gate being fitted and its pull-downs; ATP-01 shows it |
| H-02 | Outputs stay live after the host is lost | Backend crash, cable pulled, PC asleep | Failsafe: no valid command for 5 s closes the gate, zeroes the outputs and disarms; the host returning does not re-arm | SAF-03, LINK-04 | Critical | Up to 5.25 s live after the host is lost |
| H-03 | Outputs go live without an operator deciding to | Host reconnects, a script, a stray command, an observer at the dashboard | Explicit ARM, confirmed on the dashboard; output commands refused unless ARMED; observer role cannot arm; every action audited | SAF-02, SEC-01, SEC-03 | Critical | Anything with access to the serial port can send ARM (protocol.md) |
| H-04 | A corrupted command sets the wrong output | Bytes garbled between UART and host | CRC-16 on every line; a bad line refused and counted, and does not keep the failsafe away; protocol version handshake | LINK-01, LINK-02 | Critical | Unframed commands from a terminal are unchecked |
| H-05 | The firmware hangs with outputs live | A blocking call, a deadlock, a fault in a library | No blocking I/O on the loop path (the ADC hang fixed in 4.5.9); loop watchdog resets the chip; heartbeat for an external watchdog that closes the gate | SAF-06, MEAS-01 | Critical | The external watchdog is recommended, not fitted; the reset path is not yet shown on hardware (ATP-06) |
| H-06 | An output does not do what was commanded | Gate stuck or failed, wire off, a setpoint stage saturated | Gate loopback and setpoint readback, each a critical fault that disarms | OUT-01, OUT-02, OUT-03 | Critical | cone_1 and cone_2 are not read back without a second ADS1115 |
| H-07 | The switch runs at the wrong rate | A period rounded to the nearest the hardware can do; a clock misconfigured | Exact-or-refuse arithmetic; clocks checked at power-on (critical fault); edge count checked with the loopback | SW-01, SW-02, BIT-01, OUT-01 | Marginal | The interrupt path above 1 ms has microseconds of jitter |
| H-08 | Auto control drives the rig somewhere harmful | A poorly fitted model; inputs unlike the training data | Runtime-assurance guard: validated models only, envelope and step limit, hold on drift, operator override (ADR 0007) | AUTO-01, AUTO-02, AUTO-03, AUTO-04 | Critical | The guard's thresholds are engineering judgement, not derived from the rig |
| H-09 | Someone on the network operates the rig | Dashboard exposed beyond the PC | No serving off loopback without a password and TLS; observer role; deployment build has no radio | SEC-01, SEC-02, SEC-05 | Critical | The PC itself is the boundary |
| H-10 | A bad or malicious firmware image runs | A broken update; a tampered image | Updates only while disarmed; an updated image kept only if its self-test passes; no radio in the deployment build; SBOM and pinned versions | SAF-07, SEC-06, SEC-05, SEC-07, SEC-08 | Critical | No secure boot or flash encryption: anyone with the USB port can flash anything |
| H-11 | The model learns from bad data | Simulated readings mixed in; corrupted or non-numeric readings | Simulated readings tagged and kept out of training; CRC; non-finite readings stored as missing; provenance on every dataset and model | DATA-01, LINK-01, SEC-04 | Marginal | Real but unrepresentative data is not detected until drift |
| H-12 | The board resets while armed and carries on | Brownout, panic, watchdog | Comes back in FAULT; ARM refused until the operator clears it | SAF-05, SAF-04 | Critical | |
| H-13 | The loop runs slow and delays the failsafe and readings | Blocking serial output, a slow I²C bus | Loop pass timed against a 20 ms budget (a fault if over); serial output buffered (timing-budget.md) | BIT-02 | Marginal | Measured only on the bench (ATP-07) |

## Failure modes of the controller's own parts

| Part | Failure mode | Effect | Detected by | Response | Requirements |
|---|---|---|---|---|---|
| ADS1115 | Absent or drops off the bus | No readings; before 4.5.9 the loop froze | Conversion timeout (40 ms) | `ADC_LOST`, readings stop, everything else carries on | MEAS-01, BIT-02 |
| ADS1115 | Reads wrong | Model and operator misled | Not detected by the board | Bench calibration (ATP-08) | MEAS-03 |
| AND gate | Output stuck high | Switch reaches the load while disarmed | Gate loopback | `GATE_MISMATCH`, critical | OUT-01 |
| AND gate | Output stuck low | Switch never reaches the load | Gate loopback | `GATE_MISMATCH` or `SWITCH_FREQUENCY`, critical | OUT-01 |
| LEDC or timer | Fails to start, or a period change does not take | Wrong switch rate or none | Power-on self-test; readback of the LEDC settings; edge count | `SWITCH_GENERATOR` or `SWITCH_FREQUENCY`, critical; `ERROR: Switch timer unavailable!` | BIT-01, OUT-01 |
| Clocks | APB not 80 MHz, or REF_TICK divider changed | Every switch period wrong | Power-on self-test | `CLOCK_CONFIG`, critical | BIT-01 |
| Setpoint stage | Saturated, disconnected | Setpoint differs from command | Readback (setpoints 1-3) | `SETPOINT_MISMATCH`, critical | OUT-02 |
| Heap or stack | Exhausted | Crash | Floors checked every second | `LOW_MEMORY` | BIT-02 |
| Main loop | Hangs | Failsafe and heartbeat stop | Loop watchdog; external watchdog | Chip reset, comes up in FAULT | SAF-06, SAF-05 |
| Main loop | Runs slow | Commands, readings, failsafe delayed | Pass timed every pass | `LOOP_OVERRUN` | BIT-02 |
| Serial link | Bytes corrupted | Wrong command or reading | CRC | Line refused and counted | LINK-01 |
| Serial link | Lost | Board unattended | Failsafe on the board; missing replies on the host | Disarm after 5 s | SAF-03, LINK-03 |
| Serial link | Flooded | Commands delayed | Overflow counted | Line discarded, board keeps answering | LINK-05 |
| Host PC | Backend crashes | Board unattended | Failsafe | Disarm after 5 s | SAF-03 |
| Power | Brownout | Reset mid-run | Reset reason | Comes back in FAULT | SAF-05 |
| Model | Poor fit, or inputs drift | Bad proposals | Validation score; drift check | Auto control refused or held | AUTO-01, AUTO-03 |

## What this does not cover

- The rig beyond the controller's outputs: supplies, interlocks, the beam line. Those need the
  rig's own hazard analysis, which should treat this controller as one input.
- Common-cause failures of the PC and the board sharing a supply.
- Physical access: the USB port can reflash the board and send any command.
