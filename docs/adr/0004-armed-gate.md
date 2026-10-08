# 0004 An AND gate, enabled by a separate ARMED line, between GPIO 16 and the load

Status: accepted, 4.6.2. When ARMED rises was changed by ADR 0005.

## Context

GPIO 16 is undriven from reset, and before any of this firmware runs the Arduino core
probes for PSRAM, during which GPIO 16 is the PSRAM chip-select (`CONFIG_D0WD_PSRAM_CS_IO=16`)
and idles high. Firmware cannot stop that. Moving the switch to another pin was the
alternative.

## Decision

The switch reaches its load through an AND gate (74LVC1G08 on 3.3 V, or 74AHCT1G08 on 5 V for
a 5 V load). Its second input is ARMED, GPIO 23, with a 10 kΩ pull-down; the gate output has
its own pull-down. The firmware drives ARMED low before anything else in `setup()` and
drops it before anything else in every path to safe.

## Reasons

- Nothing GPIO 16 does before the firmware is in control can reach the load.
- GPIO 23 is not a strapping pin and neither the bootloader nor the PSRAM probe touches it.
- The safe state gets two independent paths to low: the switch line and the gate.
- An external watchdog can be ANDed in too (a 3-input 74LVC1G11), closing the gate if the
  loop stops, independent of the ESP32.

## Consequences

- One part and two resistors on the board. The wiring is in the README.
- What the gate output drives is not yet decided (README). That decides whether a driver
  or a series resistor follows the gate.
- With the gate loopback fitted (ADR 0008) the board checks the gate output follows ARMED.
