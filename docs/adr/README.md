# Architecture decision records

One record per decision that shapes the system and would be expensive to reverse: what
was decided, why, what else was considered, and what it costs. A record is not edited
once accepted; a later decision that changes it gets its own record and marks the old
one superseded.

| ADR | Decision | Status |
|---|---|---|
| [0001](0001-switch-line-on-ledc.md) | Generate the switch line with an LEDC channel, not MCPWM, RMT or an interrupt | Accepted (4.5.6) |
| [0002](0002-ref-tick-clock.md) | Clock the switch from the 1 MHz REF_TICK and only ever use whole dividers | Accepted (4.5.5) |
| [0003](0003-interrupt-path-above-1-ms.md) | Keep a timer interrupt for switch times above 1 ms | Accepted (4.5.6) |
| [0004](0004-armed-gate.md) | Put an AND gate enabled by a separate ARMED line between GPIO 16 and the load | Accepted (4.6.2) |
| [0005](0005-explicit-arm.md) | Arm only on an explicit ARM command | Accepted (4.6.4), supersedes 0004's "raised by any host command" |
| [0006](0006-crc-on-every-line.md) | Frame every serial line with a CRC-16 | Accepted (4.6.7) |
| [0007](0007-runtime-assurance-for-auto-control.md) | Put an independent, simple monitor between the model and the rig | Accepted (4.6.8) |
| [0008](0008-verification-hardware-off-by-default.md) | Compile output verification in, but switch it on per build | Accepted (4.6.6) |
