# ---------- Machine Learning Utilities ----------
# Shared preprocessing, feature engineering, evaluation,
# and stability analysis tools for the embedded controller system.
# --------------------------------------------------------------- #

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import logging

# ---------------------------------------------------------------
#                        Logging setup
# ---------------------------------------------------------------

logging.basicConfig(

    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

log = logging.getLogger("ML_Toolbox")

# ---------------------------------------------------------------
#                      Feature Scaling
# ---------------------------------------------------------------

Standard_scaler = StandardScaler()

def scale_features(data_frame, columns):
   
    missing = [col for col in columns if col not in data_frame.columns]

    if missing:
        raise ValueError(f"Missing features in the DataFrame: {missing}...")


    data = data_frame[columns].values
    scaling = Standard_scaler.fit_transform(data)

    Scaled_Data_Frame = pd.DataFrame(scaling, columns=columns)

    log.debug("Scaled features: %s", columns)

    return Scaled_Data_Frame, Standard_scaler

# ---------------------------------------------------------------
#                    RNN Data Preparation
# ---------------------------------------------------------------

def Data_Windows(data_frame, sequence_length=10, features=["voltage", "response"]):
    
    if len(data_frame) <= sequence_length:
        raise ValueError("Not enough data to form sequences...")
    
    X, y = [], []
    data = data_frame[features].values

    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length, 1])

    X_window, y_window = np.array(X), np.array(y)
    log.debug(f"Windowed data → {X.shape[0]} samples of length {sequence_length}")

    return X_window, y_window

# ---------------------------------------------------------------
#                     Evaluation Metrics
# ---------------------------------------------------------------

def compute_r2(y_true, y_pred):
    return float(r2_score(y_true, y_pred))

def compute_rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def compute_mae(y_true, y_pred):
    return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))

# ---------------------------------------------------------------
#              Stability and Signal Diagnostics
# ---------------------------------------------------------------

def signal_stability(series, data_window=20, variance_threshold=0.001):
    
    if len(series) < data_window:
        return False
    
    variance = np.var(series[-data_window:])
    stability_condition = variance < variance_threshold

    log.debug(f"Stability check → Var={variance:.6f}, Stable={stability_condition}")

    return stability_condition

def moving_average(series, data_window=5):

    if len(series) < data_window:
        return np.array(series)
    
    return np.convolve(series, np.ones(data_window)/data_window, mode="valid")

# ---------------------------------------------------------------
#               Feature Engineering Utilities
# ---------------------------------------------------------------

def add_derived_features(data_frame):
   
    data_frame = data_frame.copy()

    if "voltage" in data_frame.columns:
        data_frame["dV"] = data_frame["voltage"].diff().fillna(0)

    if "response" in data_frame.columns:
        data_frame["dResponse"] = data_frame["response"].diff().fillna(0)
        data_frame["Response_MA"] = data_frame["response"].rolling(5).mean().fillna(method="bfill")

    log.debug("Derived features added.")

    return data_frame

# ---------------------------------------------------------------
#                       Dataset Actions
# ---------------------------------------------------------------

def anomaly_filter(data_frame, columns, z_thresh=3.0):
  
    cleaned_data_frame = data_frame.copy()

    for col in columns:
        z = np.abs((cleaned_data_frame[col] - cleaned_data_frame[col].mean()) / cleaned_data_frame[col].std())
        cleaned_data_frame = cleaned_data_frame[z < z_thresh]

    log.info(f"Outlier removal complete: {len(data_frame)} → {len(cleaned_data_frame)} rows")

    return cleaned_data_frame.reset_index(drop=True)

def split_train_test(data_frame, desired_tt_ratio=0.8):

    idx = int(len(data_frame) * desired_tt_ratio)

    return data_frame.iloc[:idx], data_frame.iloc[idx:]

