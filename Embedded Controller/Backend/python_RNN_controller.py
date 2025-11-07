# ----------  Neural Network Controller (PyTorch RNN)  ---------- #

# Learns from the distribution from the diode output voltage 
#           data to predict optimal DAC outputs

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

DEVICE = torch.device("cuda" if torch.cuda.is_available() 
                                else "cpu")

MODEL_PATH = Path("RNN_model.pt")

SEQUENCE_LENGTH = 10
HIDDEN_SIZE = 64
LEARNING_RATE = 1e-3

INPUT_SIZE = 2  # CHANGE ONCE RIG IS BUILT
OUTPUT_SIZE = 1 # CHANGE ONCE RIG IS BUILT

# -------------------------------------------------------
#                      Logging setup
# -------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("RNN_Controller")

# -------------------------------------------------------
#                    Model Architecture 
# -------------------------------------------------------

class RNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):

        super().__init__()

        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):

        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :]) 

        return out


# -------------------------------------------------------
#                   Model Initialisation
# -------------------------------------------------------

def model_init(retrain=False):

    model = RNN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE).to(DEVICE)

    if not retrain and MODEL_PATH.exists():
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        log.info("Loaded existing trained model.")

    else:
        log.info("Initialized new model weights (fresh start).")

    return model



model = model_init()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()

# Global scaler for consistent preprocessing
scaler = StandardScaler()

# -------------------------------------------------------
#                   Data Preparation
# -------------------------------------------------------

def prepare_sequences(data_frame, sequence_length=SEQUENCE_LENGTH):
    
    data_frame = data_frame.dropna(subset=["voltage", "response"])
    data = scaler.fit_transform(data_frame[["voltage", "response"]].values)

    X, y = [], []

    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length, 1])

    X = torch.tensor(np.array(X), dtype=torch.float32).to(DEVICE)
    y = torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(-1).to(DEVICE)

    return X, y

# -------------------------------------------------------
#                      Model Training
# -------------------------------------------------------

def train_model(data_frame, number_of_epochs=10):

    global model, optimizer
    model.train()

    try:
        X, y = prepare_sequences(data_frame)
    except ValueError:
        log.warning("Insufficient data for training...")
        return

    for epoch in range(number_of_epochs):
        optimizer.zero_grad()

        preds = model(X)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            r2 = r2_score(y.cpu().numpy(), preds.cpu().numpy())
        log.info(f"Epoch {epoch+1}/{number_of_epochs} | Loss={loss.item():.6f} | R²={r2:.3f}")

    torch.save(model.state_dict(), MODEL_PATH)
    log.info(f"Model saved to {MODEL_PATH}")

# -------------------------------------------------------
#                    Model Prediction
# -------------------------------------------------------

def model_prediction(voltage_seq, response_seq):
   
    model.eval()

    with torch.no_grad():
    
        seq = np.stack([voltage_seq, response_seq], axis=1)
        seq_scaled = scaler.transform(seq)
        seq_tensor = torch.tensor(seq_scaled, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        pred = model(seq_tensor)

    return float(pred.cpu().item())

# -------------------------------------------------------
#                    Online Updates 
# -------------------------------------------------------

def online_update(df_new):
    
    model.train()

    try:
        X, y = prepare_sequences(df_new)
    except ValueError:
        return

    optimizer.zero_grad()
    preds = model(X)
    loss = criterion(preds, y)
    loss.backward()
    optimizer.step()

    torch.save(model.state_dict(), MODEL_PATH)
    log.info(f"Online update done | Loss={loss.item():.6f}")


# if __name__ == "__main__":
    df = pd.read_csv("DataLog_latest.csv")
    df.rename(columns={"raw_message": "response"}, inplace=True)
    train_model(df, epochs=20)

    v_hist = np.linspace(0, 3.3, SEQ_LEN)
    r_hist = np.sin(v_hist) + np.random.normal(0, 0.05, SEQ_LEN)
    next_v = predict_next(v_hist, r_hist)
    log.info(f"Predicted optimal voltage: {next_v:.3f} V")
