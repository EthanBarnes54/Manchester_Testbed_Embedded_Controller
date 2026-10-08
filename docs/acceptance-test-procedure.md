# Acceptance test procedure

The bench steps that verify the requirements host tests cannot: what the board does on
the chip, and what its pins do. Each step is numbered ATP-NN. The requirements marked
"Bench ATP-NN" in [requirements.md](requirements.md) refer to these numbers.

Every step has an automated part, a manual part, or both:

- **Automated** parts are in the hardware-in-the-loop suite, `Embedded Controller/Backend/hil/`
  ([how to run it](../Embedded%20Controller/Backend/hil/README.md)). Run the whole suite once
  per session and record its summary line. Each step names the tests that cover it.
- **Manual** parts need a scope, a meter or a hand on the power switch.

A build is accepted when every step passes on it. Record each run in the step's results
table. A failure is a result, not something to retry until it passes: record it, with the
scope capture or log.

## Before a session

| Item | Record |
|---|---|
| Firmware | The `VERSION` reply: `firmware=` (git describe), `build=` (environment), `gate_loopback=`, `setpoint_readback=` |
| Board | ESP32 module and board revision, the switch output stage fitted (gate part, pull-downs, external watchdog if any) |
| Load | What the gate output drives in this session, or "none" |
| Equipment | Scope (4 channels, at least 100 MHz bandwidth for the 1 µs edges), multimeter, bench supply with adjustable voltage for ATP-13 |
| Host | Backend revision (`git describe`), Python version, OS |

Flash with `pio run -e <environment> -t upload` from `Embedded Controller/`. Accept the
environment that will be deployed (`esp32_deploy` for the rig), except where a step says
otherwise.

Scope channels used throughout:

| Channel | Signal |
|---|---|
| 1 | GPIO 16, switch_logic |
| 2 | GPIO 23, ARMED |
| 3 | The AND gate output, Y |
| 4 | Whatever the step names (a setpoint pin, GPIO 18 heartbeat, the load) |

## ATP-01 Power-on state

Verifies SW-03, SAF-01.

Automated: `hil/test_rig.py::test_after_a_reset_the_board_is_safe_and_its_self_test_passes`.

1. Scope channels 1-3, single-shot, triggered on any rising edge of channel 3, with 2 s
   capture. Press EN.
2. Repeat with a power cycle instead of EN.
3. Repeat five times each.

Pass: channel 3 (the gate output) never rises. Channel 2 (ARMED) stays low throughout.
Channel 1 may show a pulse during boot (the PSRAM probe drives GPIO 16 before any firmware
runs); record its width, since it is what the gate is there to block. `FAULTS` reports
`mode=SAFE` after each reset.

| Date | Build | Tester | Result | Notes (GPIO 16 boot pulse width) |
|---|---|---|---|---|
| | | | | |

## ATP-02 Switch timing

Verifies SW-01.

Automated: `test_every_switch_time_in_range_is_accepted_and_one_outside_is_refused`, and with
the loopback fitted `test_with_the_loopback_fitted_the_gate_switches_at_the_commanded_rate`.

`ARM`, then for each setting measure channel 3 with infinite persistence, idle and again
with a serial burst arriving (a script sending `PING` in a tight loop) and, on a build with
Wi-Fi, while associating:

| Setting (`SWITCH_PERIOD_US`) | Expected |
|---|---|
| 1 | 500 kHz, 1.000 µs high and low |
| 5 | 100 kHz |
| 50 | 10 kHz |
| 1000 | 500 Hz, the last period on the LEDC |
| 1001 | Interrupt-timed: correct on average; single edges may move by microseconds |
| 2000000 | 0.25 Hz, 2 s per level |

Pass: up to 1000, period and duty within the scope's own accuracy and no visible edge
spread; at 1001 and above, the mean period within 0.1% and every edge within 50 µs of
its place. Rise and fall time at the load recorded at 1 and 5.

| Date | Build | Tester | Result | Notes (edge times at the load) |
|---|---|---|---|---|
| | | | | |

## ATP-03 Changing the switch time

Verifies SW-04.

1. `ARM`, then `SWITCH_PERIOD_US 5`, `7`, `1000`, `1001`, `5`, `1`, capturing channel 3
   single-shot around each change with a pulse-width trigger set below the shorter period.
2. `PIN 6 1`, then `PIN 6 0`.
3. `SWITCH_PERIOD_US 0`.

Pass: at each change the line goes low at once, stays low for at least one whole new
cycle, then starts with a full high half-cycle. No pulse is shorter than the shorter of
the two settings or longer than the longer, including at 1 µs. Step 2 gives a steady high
then low; step 3 a steady low with `ACK SWITCH_PERIOD_US 0 (disabled)`. No setting from 1 to
1000 is ever answered `ERROR: Switch timer unavailable!`.

| Date | Build | Tester | Result | Notes |
|---|---|---|---|---|
| | | | | |

## ATP-04 Failsafe

Verifies SAF-03.

Automated: `test_the_failsafe_disarms_5_s_after_the_last_command_and_the_host_returning_does_not_rearm`,
`test_corrupted_commands_do_not_keep_the_failsafe_away`.

1. `ARM`, `TARGETS 1 1 1 1 1`, `SWITCH_PERIOD_US 1`. Scope channels 2 and 3, and channel 4 on
   GPIO 25 (squeeze_plate). Stop the backend process.
2. Repeat, pulling the USB cable instead (only meaningful if the board is powered from elsewhere).
3. Restart the backend.

Pass: within 5.0-5.25 s of the last command, ARMED falls first, then the gate output and
the setpoint. `FAILSAFE outputs zeroed` is reported. After step 3 the board is SAFE and
stays so until `ARM`.

| Date | Build | Tester | Result | Notes (time to ARMED low) |
|---|---|---|---|---|
| | | | | |

## ATP-05 Arming

Verifies SAF-02.

Automated: `test_a_command_that_would_energise_an_output_is_refused_while_safe`,
`test_a_command_that_zeroes_an_output_is_always_accepted`, `test_arm_and_disarm`.

1. Disarmed, send `TARGETS 1 1 1 1 1`, `PIN 3 512`, `SWITCH_PERIOD_US 5`; meter GPIO 25-27.
2. `ARM`; repeat step 1.
3. `DISARM`.
4. On the dashboard: press ARM, cancel the confirmation; press it again and confirm.

Pass: step 1 is refused with `ERROR: Not armed!` and nothing moves. Step 2 drives the
outputs. Step 3 drops ARMED and zeroes every output. Step 4 arms only on confirmation.

| Date | Build | Tester | Result | Notes |
|---|---|---|---|---|
| | | | | |

## ATP-06 Watchdogs

Verifies SAF-06.

1. Scope GPIO 18 (heartbeat) on channel 4 for 10 s.
2. With an external watchdog fitted, confirm its output stays high, then hold the
   board in reset (EN low) and confirm the watchdog output falls within its timeout and
   closes the gate.

Pass: GPIO 18 toggles on every loop pass with no gap longer than the 20 ms loop budget.

Not yet possible: proving that a hung loop resets the chip needs a test build with a
command that hangs the loop on purpose. None exists. Until one does, the reset half of
SAF-06 rests on the contract test that `enableLoopWDT()` runs at the end of setup and the
core's configuration (`CONFIG_ESP_TASK_WDT_PANIC=y`). Record this step as partial.

| Date | Build | Tester | Result | Notes |
|---|---|---|---|---|
| | | | | |

## ATP-07 Built-in test and the loop budget

Verifies BIT-01, BIT-02, BIT-03, LINK-06.

Automated: `test_the_self_test_and_health_report_answer_on_demand`,
`test_the_loop_stays_inside_its_budget_and_memory_holds`,
`test_the_backends_status_poll_does_not_hold_up_the_loop`.

1. After a reset, record the power-on `SELFTEST` line.
2. Disconnect the ADS1115's SDA. Reconnect it after 10 s.
3. Run the automated tests while the backend is not connected, then again with the
   dashboard open on the board and a sweep running, and record `HEALTH` from each.

Pass: step 1 is `SELFTEST PASS`. Step 2 raises `ADC_LOST` once, the board keeps
answering, and readings resume on reconnection. In step 3, `loop_peak_us` stays under
`loop_budget_us` with no overruns, `heap_min` above 32768 and `stack_free` above 1024.
This is the measurement behind "the checks do not slow the board down": record the
figures.

| Date | Build | Tester | Result | Notes (loop_max_us, loop_peak_us, heap_min) |
|---|---|---|---|---|
| | | | | |

## ATP-08 Measurement

Verifies MEAS-01, MEAS-02.

Automated: `test_readings_arrive_at_20_hz_numbered_and_without_gaps`.

1. Feed known voltages (0, 1.000, 2.000, 3.000 V) to AIN0 and compare `MEASURED` with the
   meter.
2. With a fixed voltage on AIN0, log `MEASURED` with the switch off, then at 5 µs, 4 µs and
   1 µs, several minutes each.

Pass: readings within 5 mV of the meter. In step 2 the mean and noise match the switch-off
log at every setting; at 1 µs look for an offset or slow wander (the ADS1115 aliasing at
twice its modulator rate, README).

| Date | Build | Tester | Result | Notes |
|---|---|---|---|---|
| | | | | |

## ATP-09 Output verification

Verifies OUT-01, OUT-02, SAF-04. Needs the loopback and readback hardware and a build that
declares them (`esp32_deploy`).

Automated: `test_with_the_loopback_fitted_the_gate_switches_at_the_commanded_rate`.

1. `ARM`, `SWITCH_PERIOD_US 1000`. Lift the loopback wire from GPIO 34.
2. Reconnect it, `CLEAR FAULTS`, `ARM`, `TARGETS 1 1 1 0 0`. Wait 1 s, then disconnect
   setpoint 1's readback from AIN1.
3. Try `ARM` before and after `CLEAR FAULTS`.

Pass: step 1 raises `GATE_MISMATCH` or `SWITCH_FREQUENCY` (critical), and the board goes to
FAULT with ARMED low. Step 2 raises `SETPOINT_MISMATCH` within about 2 s. In step 3 ARM is
refused until the fault is cleared and its condition has gone.

| Date | Build | Tester | Result | Notes |
|---|---|---|---|---|
| | | | | |

## ATP-10 Serial link

Verifies LINK-01, LINK-02, LINK-05.

Automated: `test_the_board_speaks_this_protocol`,
`test_every_protocol_line_from_the_board_carries_a_valid_crc`,
`test_a_corrupted_command_is_refused_and_counted`,
`test_a_flood_of_garbage_is_discarded_and_the_board_keeps_answering`.

1. Open the dashboard and confirm the link line names the firmware and `protocol ok`.

Pass: the automated tests pass and step 1 shows the expected firmware.

| Date | Build | Tester | Result | Notes |
|---|---|---|---|---|
| | | | | |

## ATP-11 Firmware update and rollback

Verifies SEC-06. Uses `esp32dev` with Wi-Fi credentials and an OTA password, since the
deployment build has no radio.

1. Upload a good image over the air while SAFE. Then try again while ARMED.
2. Upload an image that fails its own power-on checks.

Pass: step 1 is accepted while SAFE, with `OTA image confirmed by the power-on self-test`
after the restart, and refused while ARMED. Step 2 restarts, reports
`ERROR: OTA image failed its power-on self-test, rolling back!`, and comes back on the
previous image (check `VERSION`).

Not yet possible: step 2 needs a deliberately broken test image, and none is provided.
Record step 2 as not run until one is.

| Date | Build | Tester | Result | Notes |
|---|---|---|---|---|
| | | | | |

## ATP-12 Soak

Verifies QA-05.

Automated: `hil/test_soak.py`, with `TESTBED_SOAK_MINUTES=60`, once disarmed and once armed
with `TESTBED_SOAK_SWITCH_US=1`.

Pass: both runs pass: no fault, no missing reading, no overrun, no restart, and free heap
steady.

| Date | Build | Tester | Result | Notes (minutes, heap at start and end) |
|---|---|---|---|---|
| | | | | |

## ATP-13 Unexpected reset

Verifies SAF-05.

1. Power the board from the bench supply. With the board ARMED, drop the supply below the
   brownout threshold (about 2.4 V on the 3.3 V rail) for 100 ms, then restore it.
2. Send `FAULTS`, then `ARM`, then `CLEAR FAULTS`, then `ARM`.

Pass: the board comes back in FAULT with `last_reset=BROWNOUT` and `unexpected_resets`
one higher. The first ARM is refused and the second accepted. Panic and watchdog resets
take the same path in the firmware, but there is no way to cause them on demand without
a test build.

| Date | Build | Tester | Result | Notes |
|---|---|---|---|---|
| | | | | |
