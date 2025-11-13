import threading
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class MetricWindow:
    timestamps: Deque[float]
    beam_vals: Deque[float]
    control_vecs: Deque[np.ndarray]
    control_deltas: Deque[float]
    saturations: Deque[int]


class MLMetricCollector:
    """
    Lightweight live metrics accumulator for the dashboard ML tab.

    - Beam values are proxied by the backend's diode voltage stream.
    - Control vectors are the current 6-pin states read from backend.
    - Control effort uses squared step delta of the 5 analog pins.
    - Saturations count when any of the 5 analog pins hit 0 or 1023.

    If a controller/pipeline registers richer signals (features, scores), they can
    call `push_features` and `push_controller_step` to enhance the snapshot.
    """

    def __init__(self, maxlen: int = 2000):
        self.maxlen = int(maxlen)
        self._lock = threading.Lock()
        self._w = MetricWindow(
            timestamps=deque(maxlen=self.maxlen),
            beam_vals=deque(maxlen=self.maxlen),
            control_vecs=deque(maxlen=self.maxlen),
            control_deltas=deque(maxlen=self.maxlen),
            saturations=deque(maxlen=self.maxlen),
        )
        self._last_u: Optional[np.ndarray] = None

        self._last_features: Optional[np.ndarray] = None
        self._feature_names: Optional[List[str]] = None
        self._saliency: Optional[np.ndarray] = None

    # ---- Basic stream update (from backend polling) ----
    def update_from_backend(self, timestamp: float, beam_voltage: float, pins: List[int]):
        pins = list(pins) if pins is not None else [0, 0, 0, 0, 0, 0]
        u = np.array(pins[:5], dtype=float)

        with self._lock:
            # beam proxy
            self._w.timestamps.append(float(timestamp))
            self._w.beam_vals.append(float(beam_voltage))

            # control effort on 5 analog channels
            if self._last_u is None:
                du2 = 0.0
            else:
                du2 = float(np.sum((u - self._last_u) ** 2))
            self._last_u = u
            self._w.control_vecs.append(u)
            self._w.control_deltas.append(du2)

            # saturation count (0 or 1023)
            sat = int(np.any((u <= 0.0) | (u >= 1023.0)))
            self._w.saturations.append(sat)

    # ---- Optional richer signals from the controller/pipeline ----
    def push_features(self, features: np.ndarray, feature_names: Optional[List[str]] = None):
        with self._lock:
            self._last_features = np.array(features, dtype=float).reshape(-1)
            if feature_names is not None:
                self._feature_names = list(feature_names)

    def push_saliency(self, saliency: np.ndarray):
        with self._lock:
            self._saliency = np.array(saliency, dtype=float).reshape(-1)

    # ---- Snapshot for the dashboard ----
    def snapshot(self) -> Dict:
        with self._lock:
            t = np.array(self._w.timestamps, dtype=float)
            v = np.array(self._w.beam_vals, dtype=float)
            du2 = np.array(self._w.control_deltas, dtype=float)
            sats = np.array(self._w.saturations, dtype=int)

        if t.size == 0:
            return {
                "beam_series": [],
                "time_series": [],
                "control_effort_series": [],
                "saturations_series": [],
                "beam_mean": None,
                "beam_var": None,
                "effort_mean": None,
                "effort_max": None,
                "saturations_total": 0,
                "feature_names": self._feature_names or [],
                "last_features": self._last_features.tolist() if self._last_features is not None else [],
                "saliency": self._saliency.tolist() if self._saliency is not None else [],
            }

        # Compute over full window; caller can control window via maxlen
        beam_mean = float(np.mean(v)) if v.size else None
        beam_var = float(np.var(v)) if v.size else None
        effort_mean = float(np.mean(du2)) if du2.size else None
        effort_max = float(np.max(du2)) if du2.size else None
        sat_total = int(np.sum(sats))

        return {
            "beam_series": v.tolist(),
            "time_series": t.tolist(),
            "control_effort_series": du2.tolist(),
            "saturations_series": sats.tolist(),
            "beam_mean": beam_mean,
            "beam_var": beam_var,
            "effort_mean": effort_mean,
            "effort_max": effort_max,
            "saturations_total": sat_total,
            "feature_names": self._feature_names or [],
            "last_features": self._last_features.tolist() if self._last_features is not None else [],
            "saliency": self._saliency.tolist() if self._saliency is not None else [],
        }

