# 0007 An independent, simple monitor between the model and the rig

Status: accepted, 4.6.8.

## Context

Auto control lets the RNN set the five setpoints unattended, up to ten times a second. The
model's held-out R² has been seen at -6 on data shaped unlike a sweep, and until 4.6.8
whatever it proposed was sent. A neural network's output cannot be verified by inspection.

## Decision

`python_Autonomy_Guard.py` sits between every proposal and the board, with rules simple
enough to check by reading:

- the model may drive the rig only if its current weights earned a held-out R² of at least
  0.5;
- a proposal that is not five finite numbers is rejected;
- every proposal is held to 0-3.3 V per channel and moves each channel by at most 0.25 V per
  decision;
- while the last 2 s of inputs sit more than four training standard deviations from the
  training data, nothing new is sent;
- the fallback in every case is to hold the targets already on the rig.

## Reasons

- This is the runtime assurance (simplex) pattern: an unverifiable controller is boxed in by a
  verifiable monitor and a known-safe fallback.
- The guard imports neither torch nor the board, so it is tested on its own, and the
  mutation suite shows each rule's tests fail when the rule is removed.

## Alternatives

- Trust the model after validation. Validation says nothing about inputs unlike the training
  data, which is exactly when a model goes wrong.
- A learned safety layer. It would itself need verifying.

## Consequences

- The thresholds (0.5, 0.25 V, 4 standard deviations, 2 s) are engineering judgements, not
  derived from the rig's physics. They need revisiting once the rig's dynamics are known.
- Holding the last targets is safe only if those targets were. The board's own limits
  (ARMED, the failsafe, the output checks) remain the last line.
