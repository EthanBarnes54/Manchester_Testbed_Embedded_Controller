# 0003 Keep a timer interrupt for switch times above 1 ms

Status: accepted, 4.5.6.

## Context

The LEDC on REF_TICK covers 1-1024 µs between edges exactly (ADR 0002). The command's
range was already 2 s, and the backend and dashboard were built around it.

## Decision

The switch line is a three-state machine (`SwitchLine`: Held, Hardware, Interrupt). Up to
1000 µs the LEDC generates it; from 1001 µs to 2 s a hardware timer interrupt toggles the
pin, as before 4.5.6.

## Reasons

- Nothing the rig already relies on is taken away: the maximum, the backend and the dashboard
  are unchanged.
- At 1 ms or more between edges the interrupt costs nothing measurable, and a few
  microseconds of latency is under 1% of a half-period.

## Alternatives

- Cap the range at 1000 µs and delete the interrupt path. Simpler, with one generator fewer
  to verify, but it removes any slow gating the rig uses.

## Consequences

- Edges above 1 ms can move by microseconds, more during Wi-Fi association or flash writes.
  ATP-02 records how much.
- Two generators to verify. The power-on self-test checks both, and either failing to come
  up is the critical fault `SWITCH_GENERATOR`.
