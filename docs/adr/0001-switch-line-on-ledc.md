# 0001 Generate the switch line with an LEDC channel

Status: accepted, 4.5.6.

## Context

The switch line (GPIO 16) has to toggle with a whole-microsecond time between edges, down
to 1 µs (a 500 kHz square wave). Before 4.5.6 every edge was a timer interrupt: 20,000 a
second at the old 50 µs floor, each delayed by whatever else the CPU was doing (Wi-Fi,
serial, flash writes). That cannot reach 1 µs and cannot be exact at any speed.

The ESP32 has four peripherals that can produce a waveform without the CPU: LEDC, MCPWM,
RMT, and the I2S or SPI shift registers.

## Decision

LEDC channel 6, on high-speed timer 3, generates the line for every switch time up to
1000 µs. The CPU does no work per edge.

## Reasons

- The LEDC driver is already running for the five setpoints, so nothing new is brought up.
- The Arduino core maps channel c to timer (c / 2) % 4, so channel 6 has timer 3 to itself:
  changing the switch time cannot move the setpoint PWM frequency. A `static_assert`
  enforces this.
- `ledc_timer_set()` writes the divider register as given (confirmed by disassembling it in
  the shipped `libdriver.a`), so the period is exactly what `include/switch_timing.h`
  computes.
- With REF_TICK as its clock (ADR 0002), every whole period from 1 to 1024 µs is exact.

## Alternatives

- **MCPWM**: 160 MHz clock, any 16-bit period, and glitch-free period changes through its
  shadow registers. It needs register-level prescaler, operator and generator setup that is
  hard to get right without a scope. It is the fallback if the restart on a period change
  (below) turns out to matter.
- **RMT**: 12.5 ns resolution and 15-bit durations, but the ESP32 has no hardware loop count,
  and the wrap between repetitions would need scoping.
- **A timer interrupt for every edge**: what was there before; kept only above 1 ms (ADR 0003).

## Consequences

- A period change restarts the waveform: the line is driven low, the LEDC is primed for one
  cycle off the pin, then handed the pin by direct register writes from IRAM (4.6.1), so the
  first pulse is not stretched. The host tests check the order of every step; the scope
  check is ATP-03.
- The LEDC can only reach GPIO 16 through the GPIO matrix, which adds a fixed delay to both
  edges and leaves period and duty alone.
