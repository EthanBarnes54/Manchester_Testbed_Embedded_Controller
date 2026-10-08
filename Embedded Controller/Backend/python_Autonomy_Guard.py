from dataclasses import dataclass
import math

import numpy as np

CONTROL_CHANNELS = 5
DEFAULT_MIN_VALIDATION_R2 = 0.5
DEFAULT_MAX_STEP_V = 0.25
DEFAULT_DRIFT_Z_LIMIT = 4.0
DEFAULT_DRIFT_WINDOW = 40


@dataclass(frozen=True)
class ControlEnvelope:
    """Where each channel may be driven (volts), and how far it may move per decision."""

    minimum_v: tuple = (0.0,) * CONTROL_CHANNELS
    maximum_v: tuple = (3.3,) * CONTROL_CHANNELS
    max_step_v: float = DEFAULT_MAX_STEP_V


@dataclass
class Review:
    """What the guard decided about one proposal."""

    action: str
    targets: list | None
    reason: str = ""


@dataclass
class GuardCounts:
    accepted: int = 0
    limited: int = 0
    rejected: int = 0
    held_for_drift: int = 0


class AutonomyGuard:
    def __init__(self, envelope: ControlEnvelope | None = None, min_validation_r2: float = DEFAULT_MIN_VALIDATION_R2,
                 drift_z_limit: float = DEFAULT_DRIFT_Z_LIMIT, drift_window: int = DEFAULT_DRIFT_WINDOW):
        self.envelope = envelope or ControlEnvelope()
        self.min_validation_r2 = float(min_validation_r2)
        self.drift_z_limit = float(drift_z_limit)
        self.drift_window = int(drift_window)
        self.counts = GuardCounts()

    def model_admissible(self, validation: dict | None) -> tuple[bool, str]:
        """Whether a model with these held-out scores may drive the rig, and why not."""

        score = (validation or {}).get("validation_r2")

        if score is None:
            return False, "the model has no held-out validation score; train it with a validation split first"

        try:
            score = float(score)
        except (TypeError, ValueError):
            return False, f"the model's validation score {score!r} is not a number"

        if not math.isfinite(score) or score < self.min_validation_r2:
            return False, f"the model's held-out R2 is {score:.3f}, below the {self.min_validation_r2:.2f} floor"

        return True, ""

    def drift(self, frame, stats: dict | None) -> tuple[bool, str]:
        """Whether the recent inputs have drifted outside the training data, and which and how far."""

        if not stats or frame is None or len(frame) == 0:
            return False, ""

        recent = frame.tail(self.drift_window)
        worst_name, worst_z = None, 0.0

        for name, mean, scale in zip(stats["columns"], stats["mean"], stats["scale"]):
            if name not in recent.columns:
                continue

            values = recent[name].dropna().astype(float)

            if values.empty:
                continue

            z = abs(float(values.mean()) - float(mean)) / max(float(scale), 1e-9)

            if z > worst_z:
                worst_name, worst_z = name, z

        if worst_z > self.drift_z_limit:
            self.counts.held_for_drift += 1
            return True, f"{worst_name} is {worst_z:.1f} training standard deviations from the training data"

        return False, ""

    def review(self, proposal, current_v) -> Review:
        """Holds a proposal inside the envelope and the step limit, or rejects it."""

        try:
            proposed = np.asarray(proposal, dtype=float).reshape(-1)
            current = np.asarray(current_v, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            self.counts.rejected += 1
            return Review("reject", None, "the proposal is not numeric")

        if proposed.size != CONTROL_CHANNELS or current.size != CONTROL_CHANNELS:
            self.counts.rejected += 1
            return Review("reject", None, f"the proposal has {proposed.size} channels, not {CONTROL_CHANNELS}")

        if not np.all(np.isfinite(proposed)):
            self.counts.rejected += 1
            return Review("reject", None, "the proposal is not finite")

        low = np.asarray(self.envelope.minimum_v, dtype=float)
        high = np.asarray(self.envelope.maximum_v, dtype=float)
        step = float(self.envelope.max_step_v)
        
        limited = np.clip(np.clip(proposed, current - step, current + step), low, high)

        if np.allclose(limited, proposed, atol=1e-9):
            self.counts.accepted += 1
            return Review("accept", [float(v) for v in limited])

        self.counts.limited += 1
        worst = int(np.argmax(np.abs(limited - proposed)))
        return Review("limit", [float(v) for v in limited],
                      f"pin {worst + 1} limited from {proposed[worst]:.3f} V to {limited[worst]:.3f} V")
