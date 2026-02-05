"""
# ------------- Data Pipeline for RNN Control System ------------- #

#  Pyhton module to calculate Beam data via the backend's diode voltage 
#   stream. Control vectors utilise the current pin states read from the 
#   backend. Control effort uses squared step delta of the 5 analog pins. 
#   Saturations count when any of the 5 analog pins hit 0 or 1023.

# --------------------------------------------------------------- #
"""

from logging import log
import numpy as np
from collections import deque
from typing import Deque

#---------------------------------------------------------------#
#                     Normalisation Stage
#---------------------------------------------------------------#

class Normaliser:
    """Maintains a rolling window of pulse feature vectors to compute mean and std for normalisation."""

    def __init__(self, window_pulses: int = 50, eps: float = 1e-8, min_count: int = 10):
        """Initializes the normalizer with a specified window size, epsilon for numerical stability, and minimum count for valid statistics."""

        self.window = int(max(1, window_pulses))
        self.eps = float(eps)
        self.min_count = int(max(1, min_count))
        self._buf: Deque[np.ndarray] = deque(maxlen=self.window)

    def update(self, feature_vector: np.ndarray):
        """Adds a new feature vector to the rolling window."""

        self._buf.append(np.asarray(feature_vector, dtype=float))

    def stats(self):
        """Computes the mean and standard deviation of the feature vectors in the rolling window. Returns (mean, std) as numpy arrays. 
        If there are no vectors, returns (None, None)."""

        if not self._buf:
            log.warning("Normalizer stats requested but no data in buffer. Returning Null Vectors.")
            return None, None
        
        feature_vectors= np.stack(list(self._buf), axis=0)
        feature_mean = feature_vectors.mean(axis=0)
        feature_std = feature_vectors.std(axis=0)
        return feature_mean, feature_std

    def normalize(self, input_feature_vector: np.ndarray) -> np.ndarray:
        """Normalizes the input feature vector using the mean and std from the rolling window. 
        If there are not enough vectors in the buffer, returns a zero vector of the same dimensions."""

        feature_vector = np.asarray(input_feature_vector, dtype=float)

        if len(self._buf) < self.min_count:
            return np.zeros_like(feature_vector)
        
        mean, std = self.stats()
        std = np.where(std < self.eps, 1.0, std)

        return (feature_vector - mean) / std


def _moving_average(input_array: np.ndarray, window_size: int) -> np.ndarray:
    """Applies a moving average filter to the input array with a known window size."""

    window_size = max(1, int(window_size))

    if input_array.size == 0:
        log.warning("Moving average requested but input array is empty. Returning empty array...")    
        return input_array
    
    window_filter_weightings = np.ones(window_size, dtype=float) / float(window_size)

    return np.convolve(input_array, window_filter_weightings, mode="same")

#---------------------------------------------------------------#
#                Live Feature Calculation Stage
#---------------------------------------------------------------#
class LivePulsePipeline:
    """Processes incoming voltage data to extract features for each pulse, applying denoising and normalization."""

    def __init__(
        self,
        sampling_rate_hz: float,
        pulse_on_us: float = 10.0 * 1e-6,
        pulse_off_us: float = 10.0 * 1e-6,
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

        self.normaliser = Normaliser(window_pulses=normalization_window_pulses)
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
        """Returns the names of the features extracted for each pulse as a tuple."""

        return tuple(self._feature_names)

    def reset(self):
        """Resets the internal state of the pipeline, clearing any residual data and resetting the normaliser."""

        self._residual = np.empty(0, dtype=float)
        self.normalizer = Normaliser(window_pulses=self.normalizer.window)

    def _denoise(self, x: np.ndarray) -> np.ndarray: #BAYESIAN!?!?!?!?!???!?!?!?
        """Applies a moving average filter to the input array to reduce noise."""

        return _moving_average(np.asarray(x, dtype=float), self.denoise_samples)

    def _segment_periodic(self, signal_dataset: np.ndarray):
        """Segments the input data into pulses based on the known periodic structure of the signal.
         Returns a list of (on, off) tuples for each full pulse and any leftover data that does not form a complete pulse."""
        
        number_of_samples = signal_dataset.shape[0]
        number_of_pulses = number_of_samples // self.period_samples

        pulses = []

        for i in range(number_of_pulses):
            index = i * self.period_samples

            in_window = signal_dataset[index : index + self.pulse_on_samples]
            outside_window = signal_dataset[index + self.pulse_on_samples : index + self.period_samples]

            pulses.append((in_window, outside_window))

        remaining_samples = signal_dataset[number_of_pulses * self.period_samples :]

        return pulses, remaining_samples

    def _features_for_pulse(self, on: np.ndarray, off: np.ndarray) -> np.ndarray: #FUTURE FEATURES, RISE/FALL TIME, AREA ABOVE THRESHOLD, PEAK TO BASELINE, SKEWNESS, PULSE SYMMETRY!?!?!?!?!?
        """Calculates the features for a single pulse given the 'on' and 'off' voltage segments. Returns an array of the calculated features."""

        toggle_denoise = self._denoise(on)

        reference_data = float(off.mean()) if off.size else 0.0
        pulse_signal = toggle_denoise - reference_data

        peak_value = float(np.max(pulse_signal)) if pulse_signal.size else 0.0
        clipped_signal= np.clip(pulse_signal, a_min=0.0, a_max=None)

        current_mean = float(clipped_signal.mean()) if clipped_signal.size else 0.0
        integrated_signal = float(clipped_signal.sum() * self.time_step)
        threshold_value = self.threshold_frac_of_peak * peak_value if peak_value > 0 else np.inf

        data_above_threshold = pulse_signal >= threshold_value if np.isfinite(threshold_value) else np.zeros_like(pulse_signal, dtype=bool)

        if data_above_threshold.any():

            i = np.flatnonzero(data_above_threshold)

            width_us = (i[-1] - i[0]+ 1) * self.time_step * 1e6
            arrival_us = i[0] * self.time_step * 1e6

        else:
            width_us = 0.0
            arrival_us = 0.0

        rms_voltage = float(np.sqrt(np.mean(off ** 2))) if off.size else 0.0

        return np.array([current_mean, peak_value, integrated_signal, width_us, arrival_us, rms_voltage], dtype=float)

    def process_chunk(self, voltage_data: np.ndarray):
        """Processes a chunk of voltage data, extracting features for each pulse and normalizing them. 
        Returns a list of normalized feature vectors for the pulses in the chunk."""

        input_voltages = np.asarray(voltage_data, dtype=float)

        if input_voltages.ndim != 1:
            raise ValueError("voltages must be provided as a 1D array")
        
        data = np.concatenate([self._residual, input_voltages]) if self._residual.size else input_voltages

        if self.segmentation_mode == "periodic":
            pulses, leftover = self._segment_periodic(data)

        else:
            pulses, leftover = self._segment_periodic(data)

        self._residual = leftover
        processed_chunk = []

        for toggle_on, toggle_off in pulses:

            raw_feats = self._features_for_pulse(toggle_on, toggle_off)

            normalised_features = self.normaliser.normalise(raw_feats)
            self.normaliser.update(raw_feats)

            processed_chunk.append(normalised_features)

        return processed_chunk

    def compute_feature_vector(self, pulse_samples: np.ndarray, off_gap=None) -> np.ndarray:
        """Computes the feature vector for a single pulse given the 'on' voltage segment and an optional 'off' segment."""

        background_samples = np.asarray(off_gap, dtype=float) if off_gap is not None else np.empty(0)

        raw_feature_data = self._features_for_pulse(np.asarray(pulse_samples, dtype=float), background_samples)
        normalised_feature_vector = self.normaliser.normalise(raw_feature_data)

        self.normaliser.update(raw_feature_data)

        return normalised_feature_vector
