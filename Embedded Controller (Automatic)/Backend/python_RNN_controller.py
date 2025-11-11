# ----------  Neural Network Controller (PyTorch RNN)  ---------- #

# Learns from the distribution from the diode output voltage
# data to predict outcomes conditioned on DAC, enabling live control.

# --------------------------------------------------------------- #

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import logging
from pathlib import Path

# -------------------------------------------------------
#                 Variable Initialisation
# -------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = Path("RNN_model.pt")

SEQUENCE_LENGTH = 10
HIDDEN_SIZE = 64
LEARNING_RATE = 1e-3

# Inputs currently: [dac, voltage]; extend later with pin/perm
INPUT_SIZE = 2
OUTPUT_SIZE = 1

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


class RNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, use_elu_head: bool = True):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        if use_elu_head:
            self.head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ELU(),
                nn.Linear(hidden_size, output_size),
            )
        else:
            self.head = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.head(out[:, -1, :])
        return out


# -------------------------------------------------------
#                   Model Initialisation
# -------------------------------------------------------


def _load_checkpoint(model, scaler):
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
            # Backward compatibility: file contains state_dict only
            model.load_state_dict(ckpt)
            log.info("Loaded model state_dict (no scaler in checkpoint).")
        return True
    except Exception as e:
        log.warning(f"No valid checkpoint found: {e}")
        return False


def _save_checkpoint(model, scaler):
    payload = {
        "state_dict": model.state_dict(),
    }
    if hasattr(scaler, "mean_") and hasattr(scaler, "scale_"):
        payload["scaler"] = {
            "mean_": scaler.mean_.tolist(),
            "scale_": scaler.scale_.tolist(),
        }
    torch.save(payload, MODEL_PATH)
    log.info(f"Model checkpoint saved to {MODEL_PATH}")

scaler = StandardScaler()

def model_init(retrain: bool = False):
    model = RNN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, use_elu_head=True).to(DEVICE)
    if not retrain and MODEL_PATH.exists():
        _load_checkpoint(model, scaler)
    else:
        log.info("Initialized new model weights (fresh start).")
    return model


model = model_init()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()

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
    
    df = data_frame.dropna(subset=["dac", "voltage"]).copy()
    if len(df) < sequence_length:
        raise ValueError("Insufficient data for sequence preparation.")

    features = df[["dac", "voltage"]].values
    if fit_scaler or not _check_scaler_fitted():
        data = scaler.fit_transform(features)
    else:
        data = scaler.transform(features)

    X, y = [], []
    
    for i in range(sequence_length - 1, len(data)):
        X.append(data[i - sequence_length + 1 : i + 1])
        y.append(data[i, 1])  # voltage at time i

    X = torch.tensor(np.array(X), dtype=torch.float32).to(DEVICE)
    y = torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(-1).to(DEVICE)
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
        return

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
        log.info(
            f"Epoch {epoch+1}/{number_of_epochs} | Loss={loss.item():.6f} | R2={r2:.3f}"
        )

    _save_checkpoint(model, scaler)


# -------------------------------------------------------
#                    Model Prediction / Control
# -------------------------------------------------------


def propose_dac(data_window: pd.DataFrame, candidate_dacs: np.ndarray):
   
    if len(data_window) < SEQUENCE_LENGTH - 1:
        raise ValueError("Not enough history to propose DAC.")
    if not _check_scaler_fitted():
        raise RuntimeError("Scaler not fitted. Train the model first.")

    model.eval()
    hist = data_window.tail(SEQUENCE_LENGTH - 1)[["dac", "voltage"]].values

    last_v = data_window["voltage"].iloc[-1]
    sequences = []

    for i in candidate_dacs:
        sequence = np.vstack([hist, np.array([i, last_v], dtype=float)])
        sequences.append(sequence)

    sequences = np.stack(sequences, axis=0)

    B, T, F = sequences.shape
    flat = sequences.reshape(B * T, F)
    flat_scaled = scaler.transform(flat)
    seq_scaled = flat_scaled.reshape(B, T, F)

    with torch.no_grad():
        x = torch.tensor(seq_scaled, dtype=torch.float32).to(DEVICE)
        preds = model(x).squeeze(-1).cpu().numpy()

    best_idx = int(np.argmax(preds))
    return float(candidate_dacs[best_idx]), float(preds[best_idx])


# -------------------------------------------------------
#                    Online Updates
# -------------------------------------------------------


def online_update(new_data_frame: pd.DataFrame, grad_clip: float = 1.0):
    model.train()
    try:
    
        X, y = prepare_sequences(new_data_frame, fit_scaler=True)
    except ValueError:
        return

    optimizer.zero_grad(set_to_none=True)
    preds = model(X)
    loss = criterion(preds, y)
    loss.backward()
    if grad_clip is not None and grad_clip > 0:
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

    _save_checkpoint(model, scaler)
    log.info(f"Online update done | Loss={loss.item():.6f}")


if __name__ == "__main__":
    log.info("RNN controller module ready. No standalone execution.")
