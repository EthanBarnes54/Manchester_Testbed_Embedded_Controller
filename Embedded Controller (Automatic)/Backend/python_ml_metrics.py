
#-------------- Machine Learning Metrics Module ------------#

#   Pyhton module to collect time-series metrics from the 
#   backend controll system and to provide lightweight summaries 
#   for live output to the testbed operator via the dashboards 
#   and analysis.

# ----------------------------------------------------------#

import threading
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional

import numpy as np

#----------------------------------------------------------#
#                    Metric Data Window 
#----------------------------------------------------------#

@dataclass
class MetricWindow:

    sample_times: Deque[float]
    input_voltage: Deque[float]
    pin_controls: Deque[np.ndarray]
    pin_control_changes: Deque[float]
    saturation_indicators: Deque[int]

#----------------------------------------------------------#
#                     Metric Collector
#----------------------------------------------------------#

class MetricCollector:

    def __init__(self, maxlen: Optional[int] = None, max_retrain_length: int = 2000):

        self.max_retrain_length = int(maxlen) if maxlen is not None else int(max_retrain_length)
        self._lock = threading.Lock()
        self._window = MetricWindow(
            sample_times=deque(maxlen=self.max_retrain_length),
            input_voltage=deque(maxlen=self.max_retrain_length),
            pin_controls=deque(maxlen=self.max_retrain_length),
            pin_control_changes=deque(maxlen=self.max_retrain_length),
            saturation_indicators=deque(maxlen=self.max_retrain_length),
        )
        self._last_control_vector: Optional[np.ndarray] = None

        self._latest_features: Optional[np.ndarray] = None
        self._feature_names: Optional[List[str]] = None
        self._feature_saliency: Optional[np.ndarray] = None

    def update_stream_from_backend(self, new_sample_time: float, new_beam_voltage: float, pin_values: List[int]):
        
        pin_values = list(pin_values) if pin_values is not None else [0, 0, 0, 0, 0, 0]
        control_vector = np.array(pin_values[:5], dtype=float)

        with self._lock:
            self._window.sample_times.append(float(new_sample_time))
            self._window.input_voltage.append(float(new_beam_voltage))

            if self._last_control_vector is None:
                control_effort = 0.0
            else:
                control_effort = float(np.sum((control_vector - self._last_control_vector) ** 2))

            self._last_control_vector = control_vector
            self._window.pin_controls.append(control_vector)
            self._window.pin_control_changes.append(control_effort)

            saturation = int(np.any((control_vector <= 0.0) | (control_vector >= 1023.0)))
            self._window.saturation_indicators.append(saturation)

    def record_features(self, features: np.ndarray, feature_names: Optional[List[str]] = None):

        with self._lock:
            self._latest_features = np.array(features, dtype=float).reshape(-1)
            if feature_names is not None:
                self._feature_names = list(feature_names)

    def record_feature_saliencies(self, saliency: np.ndarray):
    
        with self._lock:
            self._feature_saliency = np.array(saliency, dtype=float).reshape(-1)

    def dashboard_snapshot(self) -> Dict:
      
        with self._lock:
            sample_times = np.array(self._window.sample_times, dtype=float)
            input_voltages = np.array(self._window.input_voltage, dtype=float)
            control_effort = np.array(self._window.pin_control_changes, dtype=float)
            saturation_ind = np.array(self._window.saturation_indicators, dtype=int)

        if sample_times.size == 0:
            return {
                "Input_Voltage_series": [],
                "Sample_Times_series": [],
                "Pin_Control_Changes_series": [],
                "Saturation_Indicators_series": [],
                "Input_Voltage_mean": None,
                "Input_Voltage_var": None,
                "Pin_Control_Changes_mean": None,
                "Pin_Control_Changes_max": None,
                "Saturation_Indicators_total": 0,
                "Feature_Names": self._feature_names or [],
                "Latest_Features": self._latest_features.tolist() if self._latest_features is not None else [],
                "Feature_Saliency": self._feature_saliency.tolist() if self._feature_saliency is not None else [],
            }

        beam_mean = float(np.mean(input_voltages)) if input_voltages.size else None
        beam_var = float(np.var(input_voltages)) if input_voltages.size else None
        effort_mean = float(np.mean(control_effort)) if control_effort.size else None
        effort_max = float(np.max(control_effort)) if control_effort.size else None
        sat_total = int(np.sum(saturation_ind))

        return {
            "Input_Voltage_series": input_voltages.tolist(),
            "Sample_Times_series": sample_times.tolist(),
            "Pin_Control_Changes_series": control_effort.tolist(),
            "Saturation_Indicators_series": saturation_ind.tolist(),
            "Input_Voltage_mean": beam_mean,
            "Input_Voltage_var": beam_var,
            "Pin_Control_Changes_mean": effort_mean,
            "Pin_Control_Changes_max": effort_max,
            "Saturation_Indicators_total": sat_total,
            "Feature_Names": self._feature_names or [],
            "Latest_Features": self._latest_features.tolist() if self._latest_features is not None else [],
            "Feature_Saliency": self._feature_saliency.tolist() if self._feature_saliency is not None else [],
        }

 


    
