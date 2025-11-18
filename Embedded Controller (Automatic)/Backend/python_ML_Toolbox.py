# -------------- Machine Learning Utilities Module -------------- #

#   Provides shared preprocessing, feature engineering, 
#   evaluation, and stability analysis for the embedded 
#   control system.

# --------------------------------------------------------------- #

import logging
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

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

standard_scaler = StandardScaler()


def scale_features(data_frame: pd.DataFrame, columns):

    missing = [col for col in columns if col not in data_frame.columns]
    if missing:
        raise ValueError(f"ERROR: Missing features in the DataFrame: {missing}!")

    data = data_frame[columns].values
    scaling = standard_scaler.fit_transform(data)
    scaled_df = pd.DataFrame(scaling, columns=columns, index=data_frame.index)
    log.debug("Scaled features: %s", columns)
    return scaled_df, standard_scaler


# ---------------------------------------------------------------
#                    RNN Data Preparation
# ---------------------------------------------------------------

def data_windows(
    data_frame: pd.DataFrame,
    sequence_length: int = 10,
    features=("voltage", "response"),
):
    
    if len(data_frame) <= sequence_length:
        raise ValueError("ERROR: Not enough data to form a data window!")

    X, y = [], []
    data = data_frame[list(features)].values
    for i in range(len(data) - sequence_length):
        X.append(data[i : i + sequence_length])
        y.append(data[i + sequence_length, 1])

    X_window, y_window = np.array(X), np.array(y)
    log.debug(
        f"Windowed data: {X_window.shape[0]} (samples of length {sequence_length})"
    )
    return X_window, y_window


# ---------------------------------------------------------------
#                     Evaluation Metrics
# ---------------------------------------------------------------

def compute_r2(y_true, y_pred) -> float:
    return float(r2_score(y_true, y_pred))


def compute_rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def compute_mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))


# ---------------------------------------------------------------
#              Stability and Signal Diagnostics
# ---------------------------------------------------------------

def signal_stability(
    series,
    data_window: int = 20,
    variance_threshold: float = 0.001,
) -> bool:
    if len(series) < data_window:
        return False
    variance = np.var(series[-data_window:])
    stability_condition = variance < variance_threshold
    log.debug(
        f"Stability check :: Var={variance:.6f}, Stable={stability_condition}"
    )
    return stability_condition


def moving_average(series, data_window: int = 5):
    if len(series) < data_window:
        return np.array(series)
    return np.convolve(series, np.ones(data_window) / data_window, mode="valid")


# ---------------------------------------------------------------
#                    Feature Engineering
# ---------------------------------------------------------------

def add_derived_features(data_frame: pd.DataFrame) -> pd.DataFrame:
    data_frame = data_frame.copy()
    if "voltage" in data_frame.columns:
        data_frame["dV"] = data_frame["voltage"].diff().fillna(0)
    if "response" in data_frame.columns:
        data_frame["dResponse"] = data_frame["response"].diff().fillna(0)
        data_frame["Response_MA"] = (
            data_frame["response"].rolling(5).mean().fillna(method="bfill")
        )
    log.debug("Derived features added to the data-frame...")
    return data_frame


# ---------------------------------------------------------------
#                       Dataset Actions
# ---------------------------------------------------------------

def anomaly_filter(
    data_frame: pd.DataFrame,
    columns,
    z_thresh: float = 3.0,
) -> pd.DataFrame:
    cleaned = data_frame.copy()
    for col in columns:
        z = np.abs((cleaned[col] - cleaned[col].mean()) / cleaned[col].std())
        cleaned = cleaned[z < z_thresh]
    log.info(
        f"Outlier removal complete: {len(data_frame)} \u27f6 {len(cleaned)} rows"
    )
    return cleaned.reset_index(drop=True)


def split_train_test(data_frame: pd.DataFrame, desired_tt_ratio: float = 0.8):
    idx = int(len(data_frame) * desired_tt_ratio)
    return data_frame.iloc[:idx], data_frame.iloc[idx:]

