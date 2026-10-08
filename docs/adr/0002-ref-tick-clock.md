# 0002 Clock the switch from REF_TICK, with whole dividers only

Status: accepted, 4.5.5.

## Context

An LEDC cycle is divider x 2^bits ticks of its source clock, with a divider of 1-1023 and a
counter of up to 20 bits. The divider register also has 8 fractional bits, but a
fractional divider produces its average by moving individual edges by a source tick.

## Decision

The switch's LEDC timer runs from REF_TICK, a 1 MHz clock derived from the 80 MHz APB.
`include/switch_timing.h` only accepts a period that a whole divider produces exactly, and
refuses any other rather than rounding it.

## Reasons

- On REF_TICK, every whole period from 1 to 1024 µs between edges has a whole divider. On
  APB the factor of 5 in 80 MHz lands in the divider, which runs out at 205 µs.
- Every edge is then exactly where it was asked for. A host test checks this for every
  period from 1 µs to 2 s on both clocks, against a brute-force search.
- At 1 µs between edges the setting is one counter bit with divider 1: high for exactly one
  of two counts, so exactly 50%.

## Consequences

- APB must stay at 80 MHz. `CONFIG_PM_ENABLE` is unset and must stay so; the power-on self-test
  checks APB and the REF_TICK divider and raises the critical fault `CLOCK_CONFIG` if either
  is wrong.
- Periods are whole microseconds. A finer step would need APB or MCPWM and a nanosecond
  command.
- The CPU runs at 240 MHz (the board definition sets it), which does not affect APB.
