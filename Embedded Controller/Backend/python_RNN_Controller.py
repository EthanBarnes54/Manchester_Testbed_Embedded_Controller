r"""
# ----------------- RNN Auto controller ----------------- #

#   Learns from diode voltage sequences to predict the next
#   diode voltage conditioned on the 5 DAC outputs, so we can
#   choose DAC targets that push the diode higher.

# Due to issues with packagE imports, if this file isnt 
# working run this in the terminal:

# C:\YOUR FOLDER LOCATION\Manchester_Testbed_Embedded_Controller>python "C:\YOUR FOLDER LOCATION\Manchester_Testbed_Embedded_Controller\Embedded Controller (Automatic)\Backend\python_RNN_Controller.py"
#python "C:\Users\b09335eb\Documents\Manchester_Testbed_Embedded_Controller\Embedded Controller (Automatic)\Backend\python_RNN_Controller.py"
# ------------------------------------------------------- #

"""



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
    "set_optimizer_type",
    "get_optimizer_type",
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
OPTIMISER_TYPE = "adam"  # Optimiser for weight updates: AdamW (decay), RMSprop (noisy), Adagrad (sparse), Adadelta (adaptive), Adamax (stable), NAdam, LBFGS, ASGD --- DROP DOPWN BOX SOON!!!

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
    """Simple GRU-based regressor with optional ELU-activated head layers for improved expressiveness."""

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

    def forward(self, input_sequence):
        """Defines the forward pass of the model. Expects input of shape (batch_size, sequence_length, input_size)"""

        model_output, _ = self.gru(input_sequence)
        predictions = self.brain(model_output[:, -1, :])
        return predictions


# -------------------------------------------------------
#                   Model Initialisation
# -------------------------------------------------------


def load_previous_weights(model, scaler):
    """Attempts to load the most recent saved model weights, matching the expected pattern. Returns True if successful, False otherwise."""
    try:
        
        model_candidates = sorted(MODEL_DIR.glob(f"{MODEL_BASENAME}*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)

        if not model_candidates:
            raise FileNotFoundError("ERROR: No model weightings found!")
        
        model_path = model_candidates[0]
        model = torch.load(model_path, map_location=DEVICE)

        if isinstance(model, dict) and "state_dict" in model:
            model.load_state_dict(model["state_dict"])
            sc = model.get("scaler", {})
            mean_ = sc.get("mean_")
            scale_ = sc.get("scale_")

            if mean_ is not None and scale_ is not None:
                scaler.mean_ = np.array(mean_, dtype=float)
                scaler.scale_ = np.array(scale_, dtype=float)

            log.info(f"Successfully loaded model weights and scalers from {model_path}...")

        else:
            model.load_state_dict(model["state_dict"])
            log.info(f"Successfully loaded model from {model_path} (no scaler in file)...")

        return True
    
    except Exception as fault:
        log.warning(f"ERROR: No valid checkpoint found or incompatible with current architecture - {fault}!")
        return False


def save_nn_weights(model, scaler):
    """Saves the current model weights and scaler state (if available) to a timestamped checkpoint file."""

    payload = {"state_dict": model.state_dict()}

    if hasattr(scaler, "mean_") and hasattr(scaler, "scale_"):
        payload["scaler"] = {"mean_": scaler.mean_.tolist(), "scale_": scaler.scale_.tolist()}

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path = MODEL_DIR / f"{MODEL_BASENAME}_{timestamp}.pt"

    torch.save(payload, path)
    torch.save(payload, MODEL_PATH)

    log.info(f"Model checkpoint saved to {path} (latest model can be found in -> {MODEL_PATH})...")


def _make_optimizer(parameters, learning_rate: float | None = None):
    """Creates a new optimiser instance based on the current learning rate and momentum settings. 
    If 'learning_rate' is provided, it overrides the current learning rate for this optimiser instance only."""

    learning_rate_value = _current_learning_rate if learning_rate is None else float(learning_rate)
    opt = str(OPTIMIZER_TYPE).lower()

    if opt == "sgd":
        return optim.SGD(parameters, lr=learning_rate_value, momentum=_current_momentum, nesterov=True)

    if opt == "adamw":
        return optim.AdamW(parameters, lr=learning_rate_value, betas=(_current_momentum, 0.999))

    if opt == "rmsprop":
        return optim.RMSprop(parameters, lr=learning_rate_value, momentum=_current_momentum)

    if opt == "adagrad":
        return optim.Adagrad(parameters, lr=learning_rate_value)

    if opt == "adadelta":
        return optim.Adadelta(parameters, lr=learning_rate_value)

    if opt == "adamax":
        return optim.Adamax(parameters, lr=learning_rate_value, betas=(_current_momentum, 0.999))

    if opt == "nadam":
        return optim.NAdam(parameters, lr=learning_rate_value, betas=(_current_momentum, 0.999))

    if opt == "lbfgs":
        return optim.LBFGS(parameters, lr=learning_rate_value)

    if opt == "asgd":
        return optim.ASGD(parameters, lr=learning_rate_value)

    return optim.Adam(parameters, lr=learning_rate_value, betas=(_current_momentum, 0.999))


scaler = StandardScaler()


def model_init(retrain: bool = False):
    """Initializes the RNN model and attempts to load previous weights if available. If 'retrain' is True, it will skip loading and return a fresh instance of the previous model."""

    model = _RNN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, use_elu_head=True).to(DEVICE)

    if not retrain and MODEL_PATH.exists():
        load_previous_weights(model, scaler)

    else:
        log.info("Reset model weights. Please perofrm a retrain before operation...")

    return model


model = model_init()
optimiser = _make_optimizer(model.parameters())
Mean_error_loss = nn.MSELoss()

def set_learning_rate(learning_rate: float) -> float:
    """Sets the learning rate for the optimiser, ensuring it falls within defined bounds. Returns the new learning rate."""
    
    global _current_learning_rate

    try:
        learning_rate = float(learning_rate)
    except Exception as fault:
        raise ValueError(f"Invalid learning rate '{learning_rate}' - {fault}") from fault
    
    learning_rate = max(MIN_LEARNING_RATE, min(MAX_LEARNING_RATE, learning_rate))
    _current_learning_rate = learning_rate

    for group in optimiser.param_groups:
        group["learning_rate"] = learning_rate

    log.info(f"Learning rate set to {learning_rate:.3e}")
    return learning_rate


def get_learning_rate() -> float:
    """Returns the current learning rate."""

    return float(_current_learning_rate)


def set_momentum(momentum: float) -> float:
    """Sets the learning momentum, ensuring it falls within valid bounds. Returns the new momentum value."""
    
    global _current_momentum, optimiser

    try:
        current_momentum = float(momentum)

    except Exception as fault:
        raise ValueError(f"ERROR- Invalid momentum - {fault}!")
    
    current_momentum = max(0.0, min(0.999, current_momentum))
    _current_momentum = current_momentum

    optimiser = _make_optimizer(model.parameters())
    log.info(f"Momentum set to {current_momentum:.3f}...")

    return _current_momentum

def get_momentum() -> float:
    """Returns the current learning momentum."""
    return float(_current_momentum)


def set_optimiser_type(optimiser_type: str) -> str:
    """Sets the optimiser type and rebuilds the optimiser instance."""

    global OPTIMISER_TYPE, optimiser
    optimiser = str(optimiser_type).lower()
    optimiser_options = {"adam", "sgd", "adamw", "rmsprop", "adagrad", "adadelta", "adamax", "nadam", "lbfgs", "asgd"}

    if optimiser not in optimiser_options:
        raise ValueError(f"ERROR: Unknown optimiser type '{optimiser_type}'.")
    
    OPTIMISER_TYPE = optimiser
    optimiser = _make_optimizer(model.parameters())
    log.info(f"Optimiser set to {OPTIMISER_TYPE}...")
    return OPTIMISER_TYPE


def get_optimiser_type() -> str:
    """Returns the currently configured optimiser type."""
    return str(OPTIMISER_TYPE)


# -------------------------------------------------------
#                   Data Preparation
# -------------------------------------------------------


def _check_scaler_fitted():
    """Checks if the scaler has been fitted by verifying the presence of 'mean_' and 'scale_' attributes."""

    return hasattr(scaler, "mean_") and hasattr(scaler, "scale_")

def prepare_sequences(data_frame: pd.DataFrame, sequence_length: int = SEQUENCE_LENGTH, fit_scaler: bool = False):
    """Prepares input sequences and target vectors from the provided DataFrame for model training or prediction. 
    If 'fit_scaler' is True, it will fit the scaler to the data; otherwise, it will use the existing scaler for transformation."""

    required_columns = [f"pin_{i}" for i in range(1, 6)] + ["voltage"]
    filtered_data_frame = data_frame.dropna(subset = required_columns).copy()

    if len(filtered_data_frame) <= sequence_length:
        raise ValueError("ERROR: Insufficient data for sequence preparation!")

    feature_matrix = filtered_data_frame[required_columns].values
    target_vector = filtered_data_frame["voltage"].values

    if fit_scaler or not _check_scaler_fitted():
        data = scaler.fit_transform(feature_matrix)

    else:
        data = scaler.transform(feature_matrix)

    input_sequences, target_values = [], []
    
    for i in range(sequence_length - 1, len(data) - 1):
        input_sequences.append(data[i - sequence_length + 1 : i + 1])
        target_values.append(target_vector[i + 1])

    if not input_sequences or not target_values:
        raise ValueError("ERROR: Could not build any training sequences! (Check data-frame and sequence length...)")

    input_sequences = torch.tensor(np.array(input_sequences), dtype=torch.float32).to(DEVICE)
    target_values = torch.tensor(np.array(target_values)[:, None], dtype=torch.float32).to(DEVICE)
    return input_sequences, target_values


# -------------------------------------------------------
#                      Model Training
# -------------------------------------------------------


def train_model(data_frame: pd.DataFrame, number_of_epochs: int = 10, grad_clip_threshold: float = 1.0, save: bool = False):
    """Trains the RNN model on the provided DataFrame for a specified number of epochs. 
    It prepares the data sequences, performs backpropagation, and optionally saves the model weights after training."""

    global model, optimiser
    model.train()

    try:
        input_sequences, target_values = prepare_sequences(data_frame, fit_scaler=True)
    except ValueError:
        log.warning("WARNING: Insufficient data for training...")
        return None

    last_loss, last_r2 = None, None

    for epoch in range(number_of_epochs):

        if isinstance(optimiser, optim.LBFGS):
            def closure():
                optimiser.zero_grad(set_to_none=True)
                preds = model(input_sequences)
                loss = Mean_error_loss(preds, target_values)
                loss.backward()
                if grad_clip_threshold is not None and grad_clip_threshold > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip_threshold)
                return loss

            loss = optimiser.step(closure)
            preds = model(input_sequences)

        else:
            optimiser.zero_grad(set_to_none=True)
            preds = model(input_sequences)

            loss = Mean_error_loss(preds, target_values)
            loss.backward()

            if grad_clip_threshold is not None and grad_clip_threshold > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip_threshold)

            optimiser.step()

        with torch.no_grad():
            r2 = r2_score(target_values.cpu().numpy(), preds.cpu().numpy())
            last_loss = float(loss.item())
            last_r2 = float(r2)

        log.info(f"Epoch {epoch+1}/{number_of_epochs} | Loss={loss.item():.6f} | R2={r2:.3f}")

    if save:
        save_nn_weights(model, scaler)

    return {"loss": last_loss, "r2": last_r2}


# -------------------------------------------------------
#                    Model Prediction / Control
# -------------------------------------------------------


def propose_control_vector(data_window: pd.DataFrame, output_mode: str = "volts", num_candidates: int = 64, change_penalty: float = 0.1):
    """Given a recent window of data, proposes a control vector (DAC settings) that is predicted to increase the diode voltage, 
    while penalizing large changes from the current state."""

    required_columns = [f"pin_{i}" for i in range(1, 6)] + ["voltage"]
    filtered_data_frame = data_window.dropna(subset=required_columns)

    if len(filtered_data_frame) < SEQUENCE_LENGTH:
        raise ValueError("ERROR: Insufficeint data for control generation...")
    if not _check_scaler_fitted():
        raise RuntimeError("ERROR: Scaler not fitted! Train the model first...")

    model.eval()
    data_history = filtered_data_frame.tail(SEQUENCE_LENGTH - 1)[required_columns].values

    last_voltage = float(filtered_data_frame["voltage"].iloc[-1])
    last_pins_pwm = filtered_data_frame[[f"pin_{i}" for i in range(1, 6)]].iloc[-1].values.astype(float)

    current_voltages = (last_pins_pwm / PWM_MAX_VALUE) * ANALOG_VREF

    number_of_candidates = max(1, int(num_candidates))
    random_number_generator = np.random.default_rng()

    random_candidates = random_number_generator.uniform(0.0, ANALOG_VREF, size=(number_of_candidates, 5))

    proposed_voltage_candidates = np.vstack([current_voltages, random_candidates])

    best_estimate = -np.inf
    best_pwm = None

    for voltage_candidate in proposed_voltage_candidates:

        voltage_candidate = np.clip(np.asarray(voltage_candidate, dtype=float), 0.0, ANALOG_VREF)
        pwm_candidate = np.clip((voltage_candidate / ANALOG_VREF) * PWM_MAX_VALUE, 0.0, PWM_MAX_VALUE)

        candidate_step = np.concatenate([pwm_candidate, [last_voltage]])
        candidate_input_sequence = np.vstack([data_history, candidate_step])

        normalised_input_sequence = scaler.transform(candidate_input_sequence)

        input_tensor = torch.tensor(normalised_input_sequence[None, ...], dtype=torch.float32).to(DEVICE)

        with torch.no_grad():
            predicted_voltages = float(model(input_tensor).cpu().numpy().reshape(-1)[0])

        estimation_error_penalty = float(np.mean(np.abs(voltage_candidate - current_voltages)))
        estimation_score = predicted_voltages - change_penalty * estimation_error_penalty

        if estimation_score > best_estimate:

            best_estimate = estimation_score
            best_pwm = pwm_candidate

    if best_pwm is None:
        raise RuntimeError("ERROR: Failed to generate a control suggestion...")

    if output_mode.lower() == "pwm":
        return best_pwm

    return (best_pwm / PWM_MAX_VALUE) * ANALOG_VREF


# -------------------------------------------------------
#                    Online Updates
# -------------------------------------------------------


def online_update(new_data_frame: pd.DataFrame, grad_clip_threshold: float = 1.0, save: bool = False):
    """Performs an online update of the model using a new batch of data. It prepares the data sequences, 
    performs a single optimization step, and optionally saves the updated model weights."""

    if not _check_scaler_fitted():
        log.warning("WARNING: Scaler not fitted yet! Skipping online update until an initial training run completes...")

        return False, None, None
    
    model.train()
    
    try:
        scaled_input_sequence, target_voltage = prepare_sequences(new_data_frame, fit_scaler=False)

    except ValueError:
        return log.error("ERROR: Unable to train model! Insufficient data for online update..."), False, None, None
    
    except Exception as fault:
        return log.error(f"ERROR: Online update data preparation failed - {fault}"), False, None, None
    
    if isinstance(optimiser, optim.LBFGS):
        def closure():
            optimiser.zero_grad(set_to_none=True)
            predictions = model(scaled_input_sequence)
            mse_losses = Mean_error_loss(predictions, target_voltage)
            mse_losses.backward()
            if grad_clip_threshold is not None and grad_clip_threshold > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip_threshold)
            return mse_losses

        mse_losses = optimiser.step(closure)
        predictions = model(scaled_input_sequence)

    else:
        optimiser.zero_grad(set_to_none=True)

        predictions = model(scaled_input_sequence)
        mse_losses = Mean_error_loss(predictions, target_voltage)

        mse_losses.backward()

        if grad_clip_threshold is not None and grad_clip_threshold > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip_threshold)

        optimiser.step()

    if save:
        save_nn_weights(model, scaler)

    try:
        with torch.no_grad():
            prediction_variance = r2_score(target_voltage.cpu().numpy(), predictions.detach().cpu().numpy())

    except Exception as fault:
        log.warning(f"WARNING: Failed to compute R2 score for online update - {fault}. Proceeding without variance... ")

        prediction_variance = None

    log.info(f"Online update Complete | Loss={mse_losses.item():.6f}" + (f" | R2={prediction_variance:.3f}" if prediction_variance is not None else ""))

    return True, float(mse_losses.item()), float(prediction_variance) if prediction_variance is not None else None


# -------------------------------------------------------
#               Feature Saliency (on demand)
# -------------------------------------------------------


def compute_feature_saliencies(data_frame: pd.DataFrame, max_samples: int = 50,num_permutations: int = 20):
    """Computes feature saliencies using a permutation-based approach. The fucntion evaluates the contribution of each input 
    feature to the model's predictions by randomly permuting feature values and measuring the impact on the predicted output. 
    Returns a dictionary containing the base mean prediction and the importance scores for each feature."""

    required_columns = [f"pin_{i}" for i in range(1, 6)] 
    filtered_data_frame = data_frame.dropna(subset=required_columns).copy()

    if len(filtered_data_frame) <= SEQUENCE_LENGTH:
        raise ValueError("Insufficient data to compute saliencies.")
    
    if not _check_scaler_fitted():
        raise RuntimeError("Scaler not fitted. Train the model first...")

    input_sequences, _ = prepare_sequences(filtered_data_frame, fit_scaler=False)
    number_of_samples = input_sequences.shape[0]
    
    if number_of_samples > max_samples:
        i = torch.randperm(number_of_samples)[:max_samples]
        permutation_sequences = input_sequences[i]

    else:
        permutation_sequences = input_sequences

    reference_sequence = torch.mean(permutation_sequences, dim=0, keepdim=True)
    features = permutation_sequences.shape[2]

    contribution_sum = torch.zeros(features, dtype=torch.float32)
    total_counts = 0

    model.eval()
    base_mean = None

    with torch.no_grad():
        base_mean_tensor = model(reference_sequence).squeeze()

        try:
            base_mean = float(torch.mean(base_mean_tensor).item())

        except Exception as fault:
            log.warning(f"WARNING: Failed to compute base mean for saliency - {fault}. Proceeding without base mean...")
            base_mean = None

        for i in range(permutation_sequences.shape[0]):
            sample_window = permutation_sequences[i : i + 1].clone()

            for _ in range(max(1, int(num_permutations))):
                test_permutations = torch.randperm(features)
                feature_subset = reference_sequence.clone()

                previous_prediciton = model(feature_subset).squeeze()

                for j in test_permutations:

                    feature_index = int(j)
                    feature_subset[:, :, feature_index] = sample_window[:, :, feature_index]

                    new_prediction = model(feature_subset).squeeze()

                    contributions = new_prediction - previous_prediciton
                    contribution_sum[feature_index] += contributions.item()

                    previous_prediciton = new_prediction

                total_counts += features

    if total_counts == 0:
        raise RuntimeError("ERROR: Failed to compute shap values (no contributions recorded)!")

    impacts = [(required_columns[i], float(contribution_sum[i] / total_counts)) for i in range(min(features, len(required_columns)))]
    impacts.sort(key=lambda x: x[1], reverse=True)

    component_names = {"pin_1": "Squeeze Plate", "pin_2": "Ion Source", "pin_3": "Wein Filter", "pin_4": "Upper Cone", "pin_5": "Lower Cone"}

    names = [component_names.get(name, name.replace("_", " ")) for name, _ in impacts]
    values = [val for _, val in impacts]

    return {"base_mean": base_mean, "feature_names": names, "importances": values,}


# -------------------------------------------------------
#                 Data Pipeline Integration
# -------------------------------------------------------


class RNNController:
    """Acts as the Data Pipeline interface for the RNN model, managing data history, model predictions, and interactions with the pipeline.
    Provides methods for processing incoming data chunks, generating predictions, and saving/loading model weights."""

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
        self.optimiser = _make_optimizer(self.model.parameters(), lr=learning_rate)
        self.Mean_error_loss = nn.MSELoss()
        self.sequence_length = SEQUENCE_LENGTH

        self.model_history_location = Path("RNN_model_pipeline.pt")
        self.data_history = deque(maxlen=self.sequence_length)
        self.pipeline = None

    def attach_pipeline(self, pipeline):
        """Attaches a data pipeline to the controller, allowing it to process incoming data chunks and generate predictions based on the pipeline's features."""
        self.pipeline = pipeline

    @staticmethod

    def build_input_vector(pins_state: np.ndarray, pipeline_features: np.ndarray) -> np.ndarray:
        """Constructs the input vector for the model by concatenating the current pin states and pipeline features."""

        pins = np.asarray(pins_state, dtype=float).reshape(-1)
        features = np.asarray(pipeline_features, dtype=float).reshape(-1)

        if pins.shape[0] != 5:
            raise ValueError("ERROR: Must be 5 voltage output pins connected!")
        
        return np.concatenate([pins, features], axis=0)

    def predict_from_history(self, sequence_history: np.ndarray) -> np.ndarray:
        """Generates a prediction from the model based on a history of input vectors."""

        sequence_history = np.asarray(sequence_history, dtype=float)
        
        if sequence_history.ndim != 2 or sequence_history.shape[1] != self.input_size:
            raise ValueError(f"ERROR: Data histories must be of (number_of_time_steps, {self.input_size}) dimensions!")
        
        if sequence_history.shape[0] < self.sequence_length:
            raise ValueError(f"ERROR: Predictions require at least {self.sequence_length} pieces of data data_history!")
        
        candidate_input_sequence = sequence_history[-self.sequence_length :][None, ...]
        self.model.eval()

        with torch.no_grad():
            history_tensor = torch.tensor(candidate_input_sequence, dtype=torch.float32).to(DEVICE)
            predicted_outputs = self.model(history_tensor).cpu().numpy().reshape(-1)

        return predicted_outputs

    def step_features(self, pins_state: np.ndarray, pipeline_features: np.ndarray):
        """Processes a single step of features by building the input vector, updating the data history, and generating a prediction if enough history is available."""

        feature_vector = self.build_input_vector(pins_state, pipeline_features)
        self.data_history.append(feature_vector)

        if len(self.data_history) < self.sequence_length:
            raise ValueError(f"ERROR: Not enough data history to make a meaningful prediction! Need at least {self.sequence_length} steps of data...")
            
        return self.predict_from_history(np.stack(list(self.data_history), axis=0))

    def data_chunk(self, pins_state: np.ndarray, voltages_chunk: np.ndarray):
        """Processes a chunk of voltage data from the pipeline, generating predictions for each set of features extracted from the chunk."""

        if self.pipeline is None:
            raise RuntimeError("ERROR: No data pipeline found! Please check the connection to the control board or the data pipeline configuration...")
        
        model_predictions = []

        for features in self.pipeline.process_chunk(voltages_chunk):
            predictions = self.step_features(pins_state, features)

            if predictions is not None:
                model_predictions.append(predictions)
            else:
                raise RuntimeError("ERROR: Failed to generate predictions from data chunk! Check the data pipeline and/or model configuration...")

        return model_predictions

    def save(self, path=None):
        """Saves the current model weights to a checkpoint file. If 'path' is provided, it saves to that location; otherwise, it uses the default model history location."""

        file_path = path if path is not None else self.model_history_location
        torch.save({"state_dict": self.model.state_dict()}, file_path)

    def load(self, path=None) -> bool:
        """Attempts to load model weights from a checkpoint file. If 'path' is provided, it loads from that location; otherwise, 
        uses the default model history location. Returns True if successful, False otherwise."""

        file_path = path if path is not None else self.model_history_location

        try:
            model_path = torch.load(file_path, map_location=DEVICE)
            state = model_path["state_dict"] if isinstance(model_path, dict) and "state_dict" in model_path else model_path
            self.model.load_state_dict(state)
            
            return True
        
        except Exception as fault:
            log.warning(f"Failed to load model from path '{file_path}' - {fault}")
            return False


if __name__ == "__main__":
    log.info("RNN controller module ready! No standalone execution possible...")
