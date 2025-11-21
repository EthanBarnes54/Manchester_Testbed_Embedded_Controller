# ----------------- RNN Auto controller ----------------- #

#   Learns from diode voltage sequences to predict the next
#   diode voltage conditioned on the 5 DAC outputs, so we can
#   choose DAC targets that push the diode higher.

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
import time
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
    "set_momentum",
    "get_momentum",
    "manual_save_model",
]

# -------------------------------------------------------
#                 Variable Initialisation
# -------------------------------------------------------

DEVICE = torch.device("cpu")
MODEL_BASENAME = "RNN_model"
MODEL_DIR = Path(".")
MODEL_PATH = MODEL_DIR / f"{MODEL_BASENAME}.pt"
SEQUENCE_LENGTH = 10
HIDDEN_SIZE = 64
LEARNING_RATE = 1e-3
MIN_LEARNING_RATE = 1e-5
MAX_LEARNING_RATE = 1e-1
MOMENTUM = 0.9
OPTIMIZER_TYPE = "adam"  # "adam" (with beta1 momentum) or "sgd"

# Inputs: [Squeeze_plate_voltage, Ion_source_voltage, Wein_filter_voltage, Upper_cone_voltage, Lower_cone_voltage,  Diode_voltage]

INPUT_SIZE = 6
OUTPUT_SIZE = 1
PWM_MAX_VALUE = 1023.0
ANALOG_VREF = 3.3
_current_learning_rate = LEARNING_RATE
_current_momentum = MOMENTUM

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
        # Pick the most recent checkpoint that matches the pattern
        candidates = sorted(
            MODEL_DIR.glob(f"{MODEL_BASENAME}*.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise FileNotFoundError("No checkpoints found.")
        ckpt_path = candidates[0]
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            model.load_state_dict(ckpt["state_dict"])
            sc = ckpt.get("scaler", {})
            mean_ = sc.get("mean_")
            scale_ = sc.get("scale_")
            if mean_ is not None and scale_ is not None:
                scaler.mean_ = np.array(mean_, dtype=float)
                scaler.scale_ = np.array(scale_, dtype=float)
            log.info(f"Loaded model and scaler from checkpoint {ckpt_path}.")
        else:
            model.load_state_dict(ckpt)
            log.info(f"Loaded model state_dict from {ckpt_path} (no scaler in checkpoint).")
        return True
    except Exception as fault:
        log.warning(f"ERROR: No valid checkpoint found or incompatible with current architecture - {fault}!")
        return False


def save_nn_weights(model, scaler):
    payload = {"state_dict": model.state_dict()}
    if hasattr(scaler, "mean_") and hasattr(scaler, "scale_"):
        payload["scaler"] = {
            "mean_": scaler.mean_.tolist(),
            "scale_": scaler.scale_.tolist(),
        }
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path = MODEL_DIR / f"{MODEL_BASENAME}_{timestamp}.pt"
    torch.save(payload, path)
    # Also update the latest symlink/file for convenience
    torch.save(payload, MODEL_PATH)
    log.info(f"Model checkpoint saved to {path} (latest -> {MODEL_PATH})...")


def manual_save_model() -> str:
    """
    Persist the current model and scaler once, when explicitly requested.
    Returns the path of the timestamped checkpoint.
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path = MODEL_DIR / f"{MODEL_BASENAME}_{timestamp}.pt"
    payload = {"state_dict": model.state_dict()}
    if hasattr(scaler, "mean_") and hasattr(scaler, "scale_"):
        payload["scaler"] = {"mean_": scaler.mean_.tolist(), "scale_": scaler.scale_.tolist()}
    torch.save(payload, path)
    torch.save(payload, MODEL_PATH)
    log.info(f"Manual save to {path} (latest -> {MODEL_PATH})")
    return str(path)


def _make_optimizer(parameters, lr: float | None = None):
    lr_value = _current_learning_rate if lr is None else float(lr)
    opt = str(OPTIMIZER_TYPE).lower()

    if opt == "sgd":
        return optim.SGD(parameters, lr=lr_value, momentum=_current_momentum, nesterov=True)
    
    return optim.Adam(parameters, lr=lr_value, betas=(_current_momentum, 0.999))


scaler = StandardScaler()


def model_init(retrain: bool = False):
    model = _RNN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, use_elu_head=True).to(DEVICE)
    if not retrain and MODEL_PATH.exists():
        load_previous_weights(model, scaler)
    else:
        log.info("Reset model weights. Please perofrm a retrain before operation...")
    return model


model = model_init()
optimizer = _make_optimizer(model.parameters())
criterion = nn.MSELoss()


def set_learning_rate(learning_rate: float) -> float:
    
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


def set_momentum(momentum: float) -> float:
    
    global _current_momentum, optimizer
    try:
        m = float(momentum)
    except Exception as fault:
        raise ValueError(f"Invalid momentum '{momentum}' - {fault}") from fault
    m = max(0.0, min(0.999, m))
    _current_momentum = m
    optimizer = _make_optimizer(model.parameters())
    log.info(f"Momentum (beta1) set to {m:.3f}")
    return m


def get_momentum() -> float:
    return float(_current_momentum)


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
    target_vector = df["voltage"].values

    if fit_scaler or not _check_scaler_fitted():
        data = scaler.fit_transform(feature_matrix)
    else:
        data = scaler.transform(feature_matrix)

    X, y = [], []
    
    for i in range(sequence_length - 1, len(data) - 1):
        X.append(data[i - sequence_length + 1 : i + 1])
        y.append(target_vector[i + 1])

    if not X or not y:
        raise ValueError("ERROR: Could not build any training sequences! (Check data-fram and sequence length...)")

    X = torch.tensor(np.array(X), dtype=torch.float32).to(DEVICE)
    y = torch.tensor(np.array(y)[:, None], dtype=torch.float32).to(DEVICE)
    return X, y


# -------------------------------------------------------
#                      Model Training
# -------------------------------------------------------


def train_model(
    data_frame: pd.DataFrame,
    number_of_epochs: int = 10,
    grad_clip: float = 1.0,
    save: bool = False,
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

    if save:
        save_nn_weights(model, scaler)
    return {"loss": last_loss, "r2": last_r2}


# -------------------------------------------------------
#                    Model Prediction / Control
# -------------------------------------------------------


def propose_control_vector(
    data_window: pd.DataFrame,
    output_mode: str = "volts",
    num_candidates: int = 64,
    change_penalty: float = 0.1,
):

    required_columns = [f"pin_{i}" for i in range(1, 6)] + ["voltage"]
    df = data_window.dropna(subset=required_columns)

    if len(df) < SEQUENCE_LENGTH:
        raise ValueError("ERROR: Insufficeint data for control generation...")
    if not _check_scaler_fitted():
        raise RuntimeError("ERROR: Scaler not fitted! Train the model first...")

    model.eval()

    history = df.tail(SEQUENCE_LENGTH - 1)[required_columns].values
    last_voltage = float(df["voltage"].iloc[-1])
    last_pins_pwm = df[[f"pin_{i}" for i in range(1, 6)]].iloc[-1].values.astype(float)
    current_voltages = (last_pins_pwm / PWM_MAX_VALUE) * ANALOG_VREF

    rng = np.random.default_rng()
    n_candidates = max(1, int(num_candidates))
    random_candidates = rng.uniform(0.0, ANALOG_VREF, size=(n_candidates, 5))
    candidates = np.vstack([current_voltages, random_candidates])

    best_score = -np.inf
    best_pwm = None

    for volts_candidate in candidates:
        volts_candidate = np.clip(np.asarray(volts_candidate, dtype=float), 0.0, ANALOG_VREF)
        pwm_candidate = np.clip((volts_candidate / ANALOG_VREF) * PWM_MAX_VALUE, 0.0, PWM_MAX_VALUE)

        candidate_step = np.concatenate([pwm_candidate, [last_voltage]])
        seq = np.vstack([history, candidate_step])
        scaled = scaler.transform(seq)
        x = torch.tensor(scaled[None, ...], dtype=torch.float32).to(DEVICE)

        with torch.no_grad():
            pred_voltage = float(model(x).cpu().numpy().reshape(-1)[0])

        delta_penalty = float(np.mean(np.abs(volts_candidate - current_voltages)))
        score = pred_voltage - change_penalty * delta_penalty

        if score > best_score:
            best_score = score
            best_pwm = pwm_candidate

    if best_pwm is None:
        raise RuntimeError("ERROR: Failed to generate a controlS...")

    if output_mode.lower() == "pwm":
        return best_pwm

    return (best_pwm / PWM_MAX_VALUE) * ANALOG_VREF


# -------------------------------------------------------
#                    Online Updates
# -------------------------------------------------------


def online_update(new_data_frame: pd.DataFrame, grad_clip: float = 1.0, save: bool = False):
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
    if save:
        save_nn_weights(model, scaler)
    try:
        with torch.no_grad():
            r2 = r2_score(y.cpu().numpy(), preds.detach().cpu().numpy())
    except Exception:
        r2 = None
    log.info(f"Online update done | Loss={loss.item():.6f}" + (f" | R2={r2:.3f}" if r2 is not None else ""))
    return True, float(loss.item()), float(r2) if r2 is not None else None


# -------------------------------------------------------
#               Feature Saliency (on demand)
# -------------------------------------------------------


def compute_feature_saliencies(
    data_frame: pd.DataFrame,
    max_samples: int = 50,
    num_permutations: int = 20,
):
    """
    Monte Carlo Shapley approximation (true SHAP formulation with random permutations).
    This is compute-heavy, so it is triggered explicitly via the dashboard.
    """
    required_columns = [f"pin_{i}" for i in range(1, 6)]  # focus only on controllable pins
    df = data_frame.dropna(subset=required_columns).copy()

    if len(df) <= SEQUENCE_LENGTH:
        raise ValueError("Insufficient data to compute saliencies.")
    if not _check_scaler_fitted():
        raise RuntimeError("Scaler not fitted. Train the model first.")

    X, _ = prepare_sequences(df, fit_scaler=False)
    n = X.shape[0]
    if n > max_samples:
        idx = torch.randperm(n)[:max_samples]
        X = X[idx]

    # Baseline coalition: mean feature value across samples
    baseline = torch.mean(X, dim=0, keepdim=True)
    feats = X.shape[2]

    contrib_sum = torch.zeros(feats, dtype=torch.float32)
    total_counts = 0

    model.eval()
    base_mean = None
    with torch.no_grad():
        base_mean_tensor = model(baseline).squeeze()
        try:
            base_mean = float(torch.mean(base_mean_tensor).item())
        except Exception:
            base_mean = None
        for i in range(X.shape[0]):
            x = X[i : i + 1].clone()
            for _ in range(max(1, int(num_permutations))):
                perm = torch.randperm(feats)
                coalition = baseline.clone()
                pred_prev = model(coalition).squeeze()
                for feat_idx in perm:
                    feat_idx = int(feat_idx)
                    coalition[:, :, feat_idx] = x[:, :, feat_idx]
                    pred_new = model(coalition).squeeze()
                    contrib = pred_new - pred_prev
                    contrib_sum[feat_idx] += contrib
                    pred_prev = pred_new
                total_counts += feats

    if total_counts == 0:
        raise RuntimeError("Failed to compute shap values (no contributions accumulated).")

    impacts = [(required_columns[i], float(contrib_sum[i] / total_counts)) for i in range(feats)]
    impacts.sort(key=lambda x: x[1], reverse=True)
    friendly_names = {
        "pin_1": "Squeeze Plate",
        "pin_2": "Ion Source",
        "pin_3": "Wein Filter",
        "pin_4": "Upper Cone",
        "pin_5": "Lower Cone",
    }
    names = [friendly_names.get(name, name.replace("_", " ")) for name, _ in impacts]
    values = [val for _, val in impacts]
    return {
        "base_mean": base_mean,
        "feature_names": names,
        "importances": values,
    }


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
        self.optimizer = _make_optimizer(self.model.parameters(), lr=learning_rate)
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
