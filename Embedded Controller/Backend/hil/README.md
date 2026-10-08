# Hardware-in-the-loop tests

The automated steps of the acceptance test procedure
([docs/acceptance-test-procedure.md](../../../docs/acceptance-test-procedure.md)), run
against a real board over its serial port. They are not part of the normal test run and
never run in CI: there is no board there.

## Before you start

- Flash the build you are accepting (`pio run -e esp32_deploy -t upload`, or the
  environment named in the procedure).
- Close anything else holding the port: the dashboard, the backend, a serial monitor.
- The tests reset the board through the USB bridge's RTS line, as esptool does. On a
  board without that wiring, set `TESTBED_HIL_RESET=0` and power-cycle it by hand first.

## Running

From `Embedded Controller/Backend`:

```bash
# Disarmed checks only: link, refusals, failsafe while SAFE, measurement rate, loop budget.
TESTBED_HIL_PORT=COM5 python -m pytest hil -v

# Everything, including the tests that arm the board and switch the gate.
TESTBED_HIL_PORT=COM5 TESTBED_HIL_ALLOW_ARM=1 python -m pytest hil -v

# The soak test (QA-05), disarmed, for an hour.
TESTBED_HIL_PORT=COM5 TESTBED_SOAK_MINUTES=60 python -m pytest hil -v -m soak

# The soak test armed and switching every 5 us.
TESTBED_HIL_PORT=COM5 TESTBED_HIL_ALLOW_ARM=1 TESTBED_SOAK_MINUTES=60 TESTBED_SOAK_SWITCH_US=5 \
    python -m pytest hil -v -m soak
```

On Windows PowerShell, set each variable first: `$env:TESTBED_HIL_PORT = "COM5"`.

## Arming

`TESTBED_HIL_ALLOW_ARM=1` lets the tests arm the board, which opens the ARMED gate and lets
the switch line reach whatever GPIO 16's AND gate drives. The tests keep every setpoint
at 0 V, but only set it once the rig's outputs are disconnected or safe to energise. Every
test leaves the board disarmed with its outputs at zero, pass or fail.

## What these do not cover

What the pins actually do (edge timing, jitter, the gate's behaviour through boot, the
setpoint voltages) needs a scope or a meter. Those steps are manual and are in the
procedure. With the gate loopback fitted (`gate_loopback=1` in the VERSION reply), the
board counts its own switch edges and the tests check that count, which covers frequency
but not jitter.
