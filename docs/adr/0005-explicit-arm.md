# 0005 Arm only on an explicit ARM command

Status: accepted, 4.6.4. Supersedes ADR 0004's rule that any host command raised ARMED.

## Context

In 4.6.2, ARMED rose at the end of setup and again before any host command ran, so the
host returning after a failsafe re-armed the rig without anyone deciding to. Any command,
including a keepalive `PING`, was enough.

## Decision

The board has three modes, SAFE, ARMED and FAULT, and starts in SAFE. Only `ARM` moves it to
ARMED, and only with no critical fault active or latched. A command that would energise an
output is refused (`ERROR: Not armed!`) unless ARMED; one that zeroes an output is always
accepted. `DISARM`, the failsafe and any critical fault return it to SAFE (or FAULT). The
dashboard asks for confirmation before sending `ARM`.

## Reasons

- Making the rig live is a decision a person takes, not a side effect of the link coming up.
- A refusal is explicit and visible, where an output silently ignored would not be.
- It matches what operators expect from test equipment: power-on safe, arm, run, disarm.

## Consequences

- The operator must ARM after every boot, failsafe and fault. Sweeps and auto control refuse
  to run unless the board is ARMED, and the backend sends `DISARM` when it stops.
- ATP-05 checks the refusals on the bench; host tests check every refusal in the firmware
  and the backend.
