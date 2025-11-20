# ----------------- RNN Auto controller ----------------- #

#   Learns from diode voltage sequences to predict outcomes 
#   conditioned on DAC, supporting live control logic and 
#   offline training.

# Due to issues with packagE imports, if this file isnt 
# working run this in the terminal:

# C:\YOUR FOLDER LOCATION\Manchester_Testbed_Embedded_Controller>python "C:\YOUR FOLDER LOCATION\Manchester_Testbed_Embedded_Controller\Embedded Controller (Automatic)\Backend\python_RNN_Controller.py"
#python "C:\Users\b09335eb\Documents\Manchester_Testbed_Embedded_Controller\Embedded Controller (Automatic)\Backend\python_RNN_Controller.py"
# ------------------------------------------------------- #

import logging
from pathlib import Path
from collections import deque

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

__all__ = [
    "RNNController",
    "train_model",
    "online_update",
    "propose_control_vector",
    "set_learning_rate",
    "get_learning_rate",
]

# -------------------------------------------------------
#                 Variable Initialisation
# -------------------------------------------------------

DEVICE = torch.device("cpu")
MODEL_PATH = Path("RNN_model.pt")
SEQUENCE_LENGTH = 10
HIDDEN_SIZE = 64
LEARNING_RATE = 1e-3
MIN_LEARNING_RATE = 1e-5
MAX_LEARNING_RATE = 1e-1

# Inputs: [Squeeze_plate_voltage, Ion_source_voltage, Wein_filter_voltage, Upper_cone_voltage, Lower_cone_voltage,  Diode_voltage]

INPUT_SIZE = 6
OUTPUT_SIZE = 5
PWM_MAX_VALUE = 1023.0
ANALOG_VREF = 3.3
_current_learning_rate = LEARNING_RATE

# -------------------------------------------------------
#                      Logging setup
# -------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

log = logging.getLogger("RNN_Controller")


# -------------------------------------------------------
#                    Model Architecture
# -------------------------------------------------------


class _RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, use_elu_head: bool = True):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)

        if use_elu_head:
           self.brain = nn.Sequential(
                nn.Linear(hidden_size, 2 * hidden_size),
                nn.ELU(),
                nn.Linear(2 * hidden_size, hidden_size),
                nn.ELU(),
                nn.Linear(hidden_size, output_size),
            )

        else:
            self.brain = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.brain(out[:, -1, :])
        return out


# -------------------------------------------------------
#                   Model Initialisation
# -------------------------------------------------------


def load_previous_weights(model, scaler):
    try:
        ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            model.load_state_dict(ckpt["state_dict"])
            sc = ckpt.get("scaler", {})
            mean_ = sc.get("mean_")
            scale_ = sc.get("scale_")
            if mean_ is not None and scale_ is not None:
                scaler.mean_ = np.array(mean_, dtype=float)
                scaler.scale_ = np.array(scale_, dtype=float)
            log.info("Loaded model and scaler from checkpoint.")
        else:
            model.load_state_dict(ckpt)
            log.info("Loaded model state_dict (no scaler in checkpoint).")
        return True
    except Exception as fault:
        log.warning(f"ERROR: No valid checkpoint found - {fault}!")
        return False


def save_nn_weights(model, scaler):
    payload = {
        "state_dict": model.state_dict(),
    }
    if hasattr(scaler, "mean_") and hasattr(scaler, "scale_"):
        payload["scaler"] = {
            "mean_": scaler.mean_.tolist(),
            "scale_": scaler.scale_.tolist(),
        }
    torch.save(payload, MODEL_PATH)
    log.info(f"Model checkpoint saved to {MODEL_PATH}...")


scaler = StandardScaler()


def model_init(retrain: bool = False):
    model = _RNN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, use_elu_head=True).to(DEVICE)
    if not retrain and MODEL_PATH.exists():
        load_previous_weights(model, scaler)
    else:
        log.info("Reset model weights. Please perofrm a retrain before operation...")
    return model


model = model_init()
optimizer = optim.Adam(model.parameters(), lr=_current_learning_rate)
criterion = nn.MSELoss()


def set_learning_rate(learning_rate: float) -> float:
    """
    Update the optimizer learning rate (bounded for stability) and return the active value.
    """
    global _current_learning_rate
    try:
        lr = float(learning_rate)
    except Exception as fault:
        raise ValueError(f"Invalid learning rate '{learning_rate}' - {fault}") from fault
    lr = max(MIN_LEARNING_RATE, min(MAX_LEARNING_RATE, lr))
    _current_learning_rate = lr
    for group in optimizer.param_groups:
        group["lr"] = lr
    log.info(f"Learning rate set to {lr:.3e}")
    return lr


def get_learning_rate() -> float:
    return float(_current_learning_rate)


# -------------------------------------------------------
#                   Data Preparation
# -------------------------------------------------------


def _check_scaler_fitted():
    return hasattr(scaler, "mean_") and hasattr(scaler, "scale_")

def prepare_sequences(
    data_frame: pd.DataFrame,
    sequence_length: int = SEQUENCE_LENGTH,
    fit_scaler: bool = False,
):

    required_columns = [f"pin_{i}" for i in range(1, 6)] + ["voltage"]
    df = data_frame.dropna(subset=required_columns).copy()

    if len(df) <= sequence_length:
        raise ValueError("Insufficient data for sequence preparation.")

    feature_matrix = df[required_columns].values
    target_matrix = df[[f"pin_{i}" for i in range(1, 6)]].values

    if fit_scaler or not _check_scaler_fitted():
        data = scaler.fit_transform(feature_matrix)
    else:
        data = scaler.transform(feature_matrix)

    X, y = [], []
    # Predict the next pin vector from the previous sequence
    for i in range(sequence_length - 1, len(data) - 1):
        X.append(data[i - sequence_length + 1 : i + 1])
        y.append(target_matrix[i + 1])

    if not X or not y:
        raise ValueError("ERROR: Could not build any training sequences.")

    X = torch.tensor(np.array(X), dtype=torch.float32).to(DEVICE)
    y = torch.tensor(np.array(y), dtype=torch.float32).to(DEVICE)
    return X, y


# -------------------------------------------------------
#                      Model Training
# -------------------------------------------------------


def train_model(
    data_frame: pd.DataFrame,
    number_of_epochs: int = 10,
    grad_clip: float = 1.0,
):
    global model, optimizer
    model.train()
    try:
        X, y = prepare_sequences(data_frame, fit_scaler=True)
    except ValueError:
        log.warning("Insufficient data for training...")
        return None

    last_loss, last_r2 = None, None
    for epoch in range(number_of_epochs):
        optimizer.zero_grad(set_to_none=True)
        preds = model(X)
        loss = criterion(preds, y)
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        with torch.no_grad():
            r2 = r2_score(y.cpu().numpy(), preds.cpu().numpy())
            last_loss = float(loss.item())
            last_r2 = float(r2)
        log.info(
            f"Epoch {epoch+1}/{number_of_epochs} | Loss={loss.item():.6f} | R2={r2:.3f}"
        )

    save_nn_weights(model, scaler)
    return {"loss": last_loss, "r2": last_r2}


# -------------------------------------------------------
#                    Model Prediction / Control
# -------------------------------------------------------


def propose_control_vector(data_window: pd.DataFrame, output_mode: str = "volts"):
    required_columns = [f"pin_{i}" for i in range(1, 6)] + ["voltage"]
    df = data_window.dropna(subset=required_columns)

    if len(df) < SEQUENCE_LENGTH:
        raise ValueError("Not enough history to propose control targets.")
    if not _check_scaler_fitted():
        raise RuntimeError("Scaler not fitted. Train the model first.")

    model.eval()
    seq = df.tail(SEQUENCE_LENGTH)[required_columns].values
    scaled = scaler.transform(seq)
    x = torch.tensor(scaled[None, ...], dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        preds = model(x).cpu().numpy().reshape(-1)

    pwm_targets = np.clip(preds, 0.0, PWM_MAX_VALUE)

    if output_mode.lower() == "pwm":
        return pwm_targets

    return (pwm_targets / PWM_MAX_VALUE) * ANALOG_VREF


# -------------------------------------------------------
#                    Online Updates
# -------------------------------------------------------


def online_update(new_data_frame: pd.DataFrame, grad_clip: float = 1.0):
    if not _check_scaler_fitted():
        log.warning("Scaler not fitted yet. Skipping online update until an initial training run completes...")
        return False, None, None
    model.train()
    try:
        X, y = prepare_sequences(new_data_frame, fit_scaler=False)
    except ValueError:
        return False, None, None
    except Exception as fault:
        log.warning(f"Online update data prep failed - {fault}")
        return False, None, None
    optimizer.zero_grad(set_to_none=True)
    preds = model(X)
    loss = criterion(preds, y)
    loss.backward()
    if grad_clip is not None and grad_clip > 0:
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    save_nn_weights(model, scaler)
    try:
        with torch.no_grad():
            r2 = r2_score(y.cpu().numpy(), preds.detach().cpu().numpy())
    except Exception:
        r2 = None
    log.info(f"Online update done | Loss={loss.item():.6f}" + (f" | R2={r2:.3f}" if r2 is not None else ""))
    return True, float(loss.item()), float(r2) if r2 is not None else None


# -------------------------------------------------------
#                 Data Pipeline Integration
# -------------------------------------------------------


class RNNController:
    def __init__(
        self,
        feature_dim: int,
        hidden_size: int = HIDDEN_SIZE,
        output_size: int = OUTPUT_SIZE,
        learning_rate: float = LEARNING_RATE,
        use_elu_head: bool = True,
    ):
        self.input_size = int(5 + feature_dim)
        self.model = _RNN(self.input_size, hidden_size, output_size, use_elu_head=use_elu_head).to(DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.sequence_length = SEQUENCE_LENGTH

        self._ckpt_path = Path("RNN_model_pipeline.pt")
        self.history = deque(maxlen=self.sequence_length)
        self.pipeline = None

    def attach_pipeline(self, pipeline):
        self.pipeline = pipeline

    @staticmethod
    def build_input_vector(pins_state: np.ndarray, pipeline_features: np.ndarray) -> np.ndarray:
        pins = np.asarray(pins_state, dtype=float).reshape(-1)
        feats = np.asarray(pipeline_features, dtype=float).reshape(-1)
        if pins.shape[0] != 5:
            raise ValueError("ERROR: Must be 5 pins!")
        return np.concatenate([pins, feats], axis=0)

    def predict_from_history(self, history_inputs: np.ndarray) -> np.ndarray:
        x = np.asarray(history_inputs, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.input_size:
            raise ValueError(f"ERROR: Data histories must be of (T, {self.input_size}) dimensions!")
        if x.shape[0] < self.sequence_length:
            raise ValueError(
                f"ERROR: Predictions require at least {self.sequence_length} pieces of data history!"
            )
        seq = x[-self.sequence_length :][None, ...]
        self.model.eval()
        with torch.no_grad():
            t = torch.tensor(seq, dtype=torch.float32).to(DEVICE)
            y = self.model(t).cpu().numpy().reshape(-1)
        return y

    def step_features(self, pins_state: np.ndarray, pipeline_features: np.ndarray):
        u = self.build_input_vector(pins_state, pipeline_features)
        self.history.append(u)
        if len(self.history) < self.sequence_length:
            return None
        return self.predict_from_history(np.stack(list(self.history), axis=0))

    def data_chunk(self, pins_state: np.ndarray, voltages_chunk: np.ndarray):
        if self.pipeline is None:
            raise RuntimeError("ERROR: No pipeline attached! Attach the pipeline first...")
        outputs = []
        for features in self.pipeline.process_chunk(voltages_chunk):
            y = self.step_features(pins_state, features)
            if y is not None:
                outputs.append(y)
        return outputs

    def save(self, path=None):
        p = path if path is not None else self._ckpt_path
        torch.save({"state_dict": self.model.state_dict()}, p)

    def load(self, path=None) -> bool:
        p = path if path is not None else self._ckpt_path
        try:
            ckpt = torch.load(p, map_location=DEVICE)
            state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
            self.model.load_state_dict(state)
            return True
        except Exception:
            return False


if __name__ == "__main__":
    log.info("RNN controller module ready! No standalone execution possible...")
