# --------------- Data Pipeline for the RNN Controller --------------------- #

    # Beam values are proxied by the backend's diode voltage stream.
    # Control vectors are the current 6-pin states read from backend.
    # Control effort uses squared step delta of the 5 analog pins.
    # Saturations count when any of the 5 analog pins hit 0 or 1023.

    # If a controller/pipeline registers richer signals (features, scores), they can
    #call `push_features` and `push_controller_step` to enhance the snapshot.
# ------------------------------------------------------------------------- #

import numpy as np
from collections import deque
from typing import Deque


class Normalizer:

    def __init__(self, window_pulses: int = 50, eps: float = 1e-8, min_count: int = 10):

        self.window = int(max(1, window_pulses))
        self.eps = float(eps)
        self.min_count = int(max(1, min_count))
        self._buf: Deque[np.ndarray] = deque(maxlen=self.window)

    def update(self, x: np.ndarray):
        self._buf.append(np.asarray(x, dtype=float))

    def stats(self):

        if not self._buf:
            return None, None
        
        arr = np.stack(list(self._buf), axis=0)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)

        return mean, std

    def normalize(self, x: np.ndarray) -> np.ndarray:

        x = np.asarray(x, dtype=float)

        if len(self._buf) < self.min_count:
            return np.zeros_like(x)
        
        mean, std = self.stats()
        std = np.where(std < self.eps, 1.0, std)

        return (x - mean) / std


def _moving_average(x: np.ndarray, w: int) -> np.ndarray:

    w = max(1, int(w))

    if x.size == 0:
        return x
    
    kernel = np.ones(w, dtype=float) / float(w)

    return np.convolve(x, kernel, mode="same")


class LivePulsePipeline:

    def __init__(
        self,
        sampling_rate_hz: float,
        pulse_on_us: float = 10.0 * 1e-6,
        pulse_off_us: float = 10.0  * 1e-6,
        denoise_window_us: float = 0.5,
        normalization_window_pulses: int = 50,
        segmentation_mode: str = "periodic",
        threshold_frac_of_peak: float = 0.5,
    ):
        self.sampling_frequency = float(sampling_rate_hz)
        self.time_step = 1.0 / self.sampling_frequency
        self.pulse_on_samples = int(round((pulse_on_us) / self.time_step))
        self.pulse_off_samples = int(round((pulse_off_us) / self.time_step))
        self.period_samples = self.pulse_on_samples + self.pulse_off_samples
        self.denoise_samples = max(1, int(round((denoise_window_us * 1e-6) / self.time_step)))
        self.segmentation_mode = segmentation_mode
        self.threshold_frac_of_peak = float(threshold_frac_of_peak)

        self.normalizer = Normalizer(window_pulses=normalization_window_pulses)
        self._residual = np.empty(0, dtype=float)
        self._feature_names = [
            "mean_current",
            "peak_current",
            "charge_au",
            "pulse_width",
            "arrival_time",
            "rms_voltage",
        ]

    @property
    def feature_names(self):
        return tuple(self._feature_names)

    @property
    def feature_dim(self):
        return len(self._feature_names)

    def reset(self):
        self._residual = np.empty(0, dtype=float)
        self.normalizer = Normalizer(window_pulses=self.normalizer.window)

    def _denoise(self, x: np.ndarray) -> np.ndarray:
        return _moving_average(np.asarray(x, dtype=float), self.denoise_samples)

    def _segment_periodic(self, data: np.ndarray):
        total = data.shape[0]
        n_full = total // self.period_samples
        pulses = []
        for k in range(n_full):
            start = k * self.period_samples
            on = data[start : start + self.pulse_on_samples]
            off = data[start + self.pulse_on_samples : start + self.period_samples]
            pulses.append((on, off))
        leftover = data[n_full * self.period_samples :]
        return pulses, leftover

    def _features_for_pulse(self, on: np.ndarray, off: np.ndarray) -> np.ndarray:
        
        on = np.asarray(on, dtype=float)
        off = np.asarray(off, dtype=float)

        on_dn = self._denoise(on)
        base = float(off.mean()) if off.size else 0.0
        sig = on_dn - base

        peak = float(np.max(sig)) if sig.size else 0.0
        pos = np.clip(sig, a_min=0.0, a_max=None)

        current_mean = float(pos.mean()) if pos.size else 0.0
        charge_au = float(pos.sum() * self.time_step)

        thr = self.threshold_frac_of_peak * peak if peak > 0 else np.inf
        above = sig >= thr if np.isfinite(thr) else np.zeros_like(sig, dtype=bool)
        if above.any():
            idx = np.flatnonzero(above)
            width_us = (idx[-1] - idx[0] + 1) * self.time_step * 1e6
            arrival_us = idx[0] * self.time_step * 1e6
        else:
            width_us = 0.0
            arrival_us = 0.0

        vrms = float(np.sqrt(np.mean(off ** 2))) if off.size else 0.0

        return np.array(
            [current_mean, peak, charge_au, width_us, arrival_us, vrms], dtype=float
        )

    def process_chunk(self, voltages: np.ndarray):
        x = np.asarray(voltages, dtype=float)
        if x.ndim != 1:
            raise ValueError("voltages must be a 1D array")

        data = np.concatenate([self._residual, x]) if self._residual.size else x

        if self.segmentation_mode == "periodic":
            pulses, leftover = self._segment_periodic(data)
        else:
            pulses, leftover = self._segment_periodic(data)

        self._residual = leftover

        outputs = []
        for on, off in pulses:
            raw_feats = self._features_for_pulse(on, off)
            norm_feats = self.normalizer.normalize(raw_feats)
            self.normalizer.update(raw_feats)
            outputs.append(norm_feats)

        return outputs

    def compute_feature_vector(self, on_pulse: np.ndarray, off_gap=None) -> np.ndarray:
        off = np.asarray(off_gap, dtype=float) if off_gap is not None else np.empty(0)
        raw = self._features_for_pulse(np.asarray(on_pulse, dtype=float), off)
        norm = self.normalizer.normalize(raw)
        self.normalizer.update(raw)
        return norm
