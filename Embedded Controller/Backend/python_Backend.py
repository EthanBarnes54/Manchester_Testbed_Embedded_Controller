# --------- Backend Communication Layer - ESP32 Cntrol System --------- #

# Handles serial communication between the ESP32 board and the Python environment.
# Provides live data streaming, thread-safe buffering, and command
# transmission. Designed for use with the data loggier, dashboard, signal
# pipeline, and RNN controller modules.

#------------------------------------------------------------------------#

# ---------------------------------------------------------------------- #
#                               Imports                                  #
# ---------------------------------------------------------------------- #

import itertools
import logging
import os
import queue
import random
import threading
import time

import numpy as np
import pandas as pd
import serial

from python_ML_Metrics import MetricCollector

try:

    from python_RNN_Controller import (
        train_model,
        online_update,
        set_learning_rate as _set_rnn_learning_rate,
        get_learning_rate as _get_rnn_learning_rate,
        set_momentum as _set_rnn_momentum,
        get_momentum as _get_model_momentum,
        manual_save_model,
        compute_feature_saliencies,
    )

except Exception:
    train_model = None
    online_update = None

    def _set_rnn_learning_rate(_value):
        return None

    def _get_rnn_learning_rate():
        return None

    def _set_rnn_momentum(_value):
        return None

    def _get_model_momentum():
        return None

    def manual_save_model():
        raise RuntimeError("ERROR: Manual save unavailable! (RNN controller import failed...)")

    def compute_feature_saliencies(_df, max_samples=200):
        raise RuntimeError("ERROR: Feature saliencies unavailable! (RNN controller import failed...)")


# ------------------------------------------------------------------------- # 
#                                 Logging                                   # 
# ------------------------------------------------------------------------- # 

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S",)
log = logging.getLogger("ESP32_Backend")


# -------------------------------------------------------------------------
#                             Configuration
# -------------------------------------------------------------------------

SERIAL_PORT = "COM4"  # Adjust for OTA
BAUD_RATE = 115200
RETRY_DELAY = 3.0
MAX_QUEUE_SIZE = 2000

OFFLINE = os.getenv("OFFLINE", "").strip() not in ("", "0", "false", "False")

# Alter once board design has been finalised 

MAX_MODULATION_VALUE = 1023
MAX_CONTROL_VOLTAGE = 3.3
CONTROL_PIN_COUNT = 5
ONLINE_UPDATE_INTERVAL_SEC = 5.0
DEFAULT_UPDATE_WINDOW = 30.0
UPDATE_WINDOW_TIME = 5.0
ONLINE_WINDOW_MAX_SEC = 600.0


# ------------------------------------------------------------------------- # 
#                             Backend Object                                # 
# ------------------------------------------------------------------------- # 


class SerialBackend:
    """Serial (or simulated) backend for the ESP32 embedded controller.

    This class owns the serial connection and background threads for:

    - reading and parsing messages from the embedded firmware
    - buffer data into an in-memory DataFrame
    - send control commands
    - perform periodic online model updates and training sweeps when prompted by user
    """

    def __init__(self, port=SERIAL_PORT, baud=BAUD_RATE, status: bool = False):

        """Creates and starts the backend.

        Arguments:
            port: Serial port name (for example, "COM4").
            baud: Serial baud rate.
            status: When True, starts in offline/simulated mode.
        """

        self.port = port
        self.baud = baud
        self.serial = None

        self.alive = threading.Event()
        self.lines = queue.Queue(maxsize=MAX_QUEUE_SIZE)
        self.data_lock = threading.Lock()

        self.data_frame = pd.DataFrame(
            columns=[
                "timestamp",
                "voltage",
                "pin_1",
                "pin_2",
                "pin_3",
                "pin_4",
                "pin_5",
                "switch_logic",
                "raw_message",
            ]
        )

        self.offline = status

        self.pins = [0, 0, 0, 0, 0, 0]
        self.pins_timestamp = 0.0
        self.switch_timing = None

        self.sweep_thread = None
        self.sweep_status = {"state": "idle", "progress": 0.0, "message": ""}
        self.sweep_cancel = threading.Event()

        self.save_dataset_enabled = False
        self.last_training_time_stamp = None
        self.last_update_time_stamp = None
        self.online_window_seconds = DEFAULT_UPDATE_WINDOW
        self.online_update_period = ONLINE_UPDATE_INTERVAL_SEC
        self.online_update_enabled = True

        self.thread = threading.Thread(target=self.operating_system, daemon=True)

        self.alive.set()
        self.thread.start()

        self.online_update_thread = threading.Thread(target=self._update_manager, daemon=True)
        self.online_update_thread.start()

    # ------------------------------------------------------------------ # 
    #                     Connect / Disconnect Functions                 # 
    # ------------------------------------------------------------------ # 

    def connect(self):
        """Opens the serial connection to the ESP32.

        On connection failure, the backend switches to offline mode to avoid
        continuous reconnect attempts.
        """

        if self.offline:
            time.sleep(0.1)
            return

        while self.alive.is_set():
            try:
                log.info("Connecting to ESP32...")
                self.serial = serial.Serial(self.port, self.baud, timeout=1)
                log.info(f"Connected to ESP32 on {self.port} at {self.baud} baud...")
                return
            
            except serial.SerialException as fault:
                log.warning(f"ERROR: Connection to board failed - {fault}! Reverting to offline...")
                self.offline = True
                return

    def disconnect(self):
        """Close the serial connection (if open)."""

        if self.serial and self.serial.is_open:
            try:
                log.info("Serial closing...")
                self.serial.close()
                log.info("Serial closed!")

            except Exception:
                pass
        self.serial = None

    def set_save_dataset_enabled(self, enabled: bool):
        """Toggles dataset saving."""

        try:
            self.save_dataset_enabled = bool(enabled)

        except Exception:
            self.save_dataset_enabled = False

    def set_window_update_time(self, window_seconds: float):
        """Set the online ML learning update window length.

        Arguments:
            window_seconds: Desired window length in seconds.

        Returns:
            The clamped window length in seconds.
        """
        try:
            value = float(window_seconds)

        except Exception as fault:
            raise ValueError(f"ERROR: Invalid time window '{window_seconds}' : {fault}!")
        
        bounded = max(UPDATE_WINDOW_TIME, min(ONLINE_WINDOW_MAX_SEC, value))
        self.online_window_seconds = bounded
        log.info(f"Update window duration set to {bounded:.1f} s...")

        return bounded

    def get_window_update_time(self) -> float:
        """Retrieve the current online-learning update window length."""

        return float(self.online_window_seconds)

    def set_online_learning_rate(self, learning_rate: float):
        """Sets the RNN online-learning rate."""

        if _set_rnn_learning_rate is None:
            raise RuntimeError("ERROR: Learning rate unavailable for upload! (RNN controller import failed...)")
        return _set_rnn_learning_rate(learning_rate)

    def get_learning_rate(self):
        """Return the current RNN learning rate."""

        if _get_rnn_learning_rate is None:
            return None
        return _get_rnn_learning_rate()

    def set_model_momentum(self, momentum: float):
        """Set the RNN learning momentum."""
        if _set_rnn_momentum is None:
            raise RuntimeError("ERROR: Momentum unavailable for upload! (RNN controller import failed...)")
        return _set_rnn_momentum(momentum)

    def get_model_momentum(self):
        """Return the current learning momentum."""

        if _get_model_momentum is None:
            return None
        return _get_model_momentum()

    def save_model_parameters(self):
        """Saves RNN model parameters to machine."""

        if manual_save_model is None:
            raise RuntimeError("ERROR: Manual save unavailable! (RNN controller import failed...)")
        return manual_save_model()

    def compute_feature_importance(self, max_samples: int = 200, num_permutations: int = 20):
        """Compute feature importances/saliencies from recent backend data, using shapley permutation."""

        data_frame = self.get_data()
        return compute_feature_saliencies(data_frame, max_samples=max_samples, num_permutations=num_permutations)

    def get_online_update_config(self) -> dict:
        """Returns a snapshot of online update settings."""

        return {
            "window_seconds": float(self.online_window_seconds),
            "learning_rate": self.get_learning_rate(),
            "enabled": bool(self.online_update_enabled),
            "momentum": self.get_model_momentum(),
        }

    def get_latest_point(self):
        """Return the most recent (timestamp, voltage) pair."""
        with self.data_lock:

            if self.data_frame.empty:
                return None
            
            row = self.data_frame.iloc[-1]
            return float(row["timestamp"]), float(row["voltage"])


    # ------------------------------------------------------------------ #
    #                         Object Methods                             #
    # ------------------------------------------------------------------ #

    
    @staticmethod
    def _name_to_pin_index(pin_name: str) -> int:
        """Map a pin label (or numeric string) to a 1-based pin index (1..6).

        Returns 0 when the name/index is invalid.
        """
        if pin_name is None:
            return 0
        
        pin_number = str(pin_name).strip().lower()

        pin_mapping = {
            "squeeze_plate": 1,
            "ion_source": 2,
            "wein_filter": 3,
            "cone_1": 4,
            "cone_2": 5,
            "switch_logic": 6,
        }

        if pin_number in pin_mapping:
            return pin_mapping[pin_number]
        
        try:
            i = int(pin_number)
            return i if 1 <= i <= 6 else 0
        
        except Exception as fault:
            return log.error(f"ERROR: Invalid pin index '{pin_number}' : {fault}!")

    @staticmethod
    def _clamp_signal_pulse(value) -> int:
        """Clamp a PWM pulse command to the acceptible range."""
        try:
            signal_value = int(round(float(value)))

        except Exception as fault:
            return log.error(f"ERROR: Pulse signal out of range '{value}' : {fault}!")
        
        return max(0, min(MAX_MODULATION_VALUE, signal_value))

    @staticmethod
    def clamp_voltage(voltage: float) -> float:
        """Clamp a voltage command to the acceptable range."""
        try:
            voltage_value = float(voltage)

        except Exception as fault:
            return log.error(f"ERROR: Voltage out of range '{voltage}' : {fault}!")
        
        return max(0.0, min(MAX_CONTROL_VOLTAGE, voltage_value))

    @classmethod
    def pulse_to_voltage(cls, value: float) -> float:
        """Convert a PWM pulse to volts."""

        pulse_signal = cls._clamp_signal_pulse(value)

        return (pulse_signal / MAX_MODULATION_VALUE) * MAX_CONTROL_VOLTAGE

    @classmethod
    def voltage_to_pulse(cls, voltage: float) -> int:
        """Convert voltage to a PWM pulse."""

        voltage_output = cls.clamp_voltage(voltage)
        scaled_voltage_output = (voltage_output / MAX_CONTROL_VOLTAGE) * MAX_MODULATION_VALUE

        return cls._clamp_signal_pulse(scaled_voltage_output)

    def update_pin_values(self, pin_index: int, pin_value: int) -> bool:
        """Updates cached pin values and timestamp.

        This is used both when reading from the ESP32 and when simulating.
        """

        if not (1 <= int(pin_index) <= 6):
            log.warning(f"WARNING: Pin index out of range to send update!")
            return False
        
        clamped_pin_value = self.get_pin_value(int(pin_index), int(pin_value))

        if clamped_pin_value is None:
            log.warning(f"WARNING: Unable to clamp pin values!")
            return False
        
        self.pins[int(pin_index) - 1] = clamped_pin_value
        self.pins_timestamp = time.time()

        return True

    def _append_measurement(self, timestamp: float, voltage: float, message: str):
        """Append a measurement row to the internal DataFrame."""

        status_snapshot = list(self.pins)

        row = {
            "timestamp": float(timestamp),
            "voltage": None if voltage is None else float(voltage),
            "pin_1": status_snapshot[0],
            "pin_2": status_snapshot[1],
            "pin_3": status_snapshot[2],
            "pin_4": status_snapshot[3],
            "pin_5": status_snapshot[4],
            "switch_logic": status_snapshot[5],
            "raw_message": message,
        }

        with self.data_lock:
            self.data_frame.loc[len(self.data_frame)] = row
            self.data_frame = self.data_frame.tail(1000).reset_index(drop=True)

    def operating_system(self):
        """ Reads serial messages, parses measurement / pin snapshots, and updates
        the internal buffers. When offline, generates simulated readings.
        """

        while self.alive.is_set():

            # purely offline mode - simulate data to test functionality and conectivity
            if self.offline:

                timestamp = time.time()

                analog_snapshot = [self.pulse_to_voltage(p) for p in self.pins[:CONTROL_PIN_COUNT]]
                avg_voltage = float(np.mean(analog_snapshot)) if analog_snapshot else 0.0
                noise = (random.random() - 0.5) * 0.5
                v = max(0.0, min(MAX_CONTROL_VOLTAGE, avg_voltage + noise))
                message = f"MEASURED {v:.3f}"

                if not self.lines.full():
                    self.lines.put((timestamp, message))

                self._append_measurement(timestamp, v, message)

                time.sleep(0.05)
                continue

            if self.serial is None or not self.serial.is_open:
                self.connect()

            try:
                message = self.serial.readline().decode("utf-8", "ignore").strip()

                if not message:
                    continue

                timestamp = time.time()

                if not self.lines.full():
                    self.lines.put((timestamp, message))

                if message.startswith("MEASURED"):

                    try:
                        voltage = float(message.split()[1])

                    except (IndexError, ValueError):
                        log.warning("WARNING: Inappropriate measurement received!")
                        voltage = None

                    self._append_measurement(timestamp, voltage, message)

                elif message.startswith("PINS"):
                    try:
                        for pin_assignment in message.split()[1:]:

                            if "=" not in pin_assignment:
                                continue

                            name, value_string = pin_assignment.split("=", 1)

                            try:
                                pin_value = int(float(value_string))

                            except ValueError as fault:
                                log.error(f"ERROR: Invalid pin value received - {fault}!")
                                continue

                            pin_index = self._name_to_pin_index(name)
                            self.update_pin_values(pin_index, pin_value)

                    except Exception as fault:
                        log.error(f"ERROR: Processing of pin ouputs unsuccessful - {fault}!")
                        pass

                elif message.startswith("ACK PIN"):
                    try:
                        tokens = message.split()
                        self.update_pin_values(int(tokens[2]), int(tokens[3]))

                    except Exception as fault:
                        log.error(f"ERROR: Processing of pin input readings unsuccessful - {fault}!")
                        pass

            except serial.SerialException as serial_fault:
                log.warning(f"ERROR: Serial exception - {serial_fault}! Reconnecting...")
                self.disconnect()
                time.sleep(RETRY_DELAY)

            except Exception as fault:
                log.error(f"ERROR: Unexpected fault - {fault}! Reconnecting...")
                time.sleep(0.5)

    def _update_manager(self):
        """Background loop for periodic online model updates."""
        
        while self.alive.is_set():

            time.sleep(self.online_update_period)

            if not self.online_update_enabled or online_update is None:
                continue

            try:
                if str(self.sweep_status.get("state", "")).lower() == "running":
                    continue

            except Exception as fault:
                log.warning(f"WARNING: Sweep Status unretrievable - {fault}!")
                pass

            data_frame = self.get_data()

            if data_frame.empty:
                log.warning("WARNING: No data available for updates (data frame is empty)!")
                continue

            try:
                window_floor = time.time() - max(UPDATE_WINDOW_TIME, float(self.online_window_seconds))
            except Exception:
                window_floor = time.time() - DEFAULT_UPDATE_WINDOW

            try:
                window_data_frame = data_frame[data_frame["timestamp"] >= window_floor]

            except Exception as fault:
                window_data_frame = data_frame
                log.warning(f"WARNING: Failed to allocate data to retraining window - {fault}!")

            if window_data_frame.empty:
                log.error("ERROR: No data available for updates (windowed data frame is empty)!")
                continue

            try:
                updated, loss, r2 = online_update(window_data_frame, grad_clip=1.0)
            except Exception as fault:
                log.warning(f"ERROR: Failed to update the model training data - {fault}!")
                continue

            if updated:
                current_time = time.time()
                self.last_training_time_stamp = current_time
                self.last_update_time_stamp = current_time
                try:
                    ML_METRICS.record_training_update(current_time, loss=loss, r2=r2, source="online")
                except Exception as fault:
                    log.warning(f"WARNING: Failed to record training update metrics - {fault}!")
                    pass

    # ------------------------------------------------------------------ #
    #                               Methods                              #
    # ------------------------------------------------------------------ #

    def send_command(self, command_string: str):
        """Sends a command string to the ESP32."""
        try:
            if self.offline:

                pin_assignments = command_string.strip().split()

                if not pin_assignments:
                    return

                command = pin_assignments[0].upper()

                if command == "PIN" and len(pin_assignments) == 3:
                    try:
                        command_pin_index = self._name_to_pin_index(pin_assignments[1])
                        command_pin_value = self._clamp_signal_pulse(pin_assignments[2])

                        if self.update_pin_values(command_pin_index, command_pin_value):
                            log.info(f"[SIMULATED] PIN {command_pin_index} set to {self.pins[command_pin_index-1]}")
                        
                        else:
                            log.warning(f"[SIMULATED] PIN index out of range: {command_pin_index}")

                    except Exception as fault:
                        log.error(f"[SIMULATED] Invalid PIN command: {command_string} - {fault}!")

                elif command == "TARGETS" and len(pin_assignments) >= CONTROL_PIN_COUNT + 1:
                    try:
                        raw_values = [float(token) for token in pin_assignments[1 : CONTROL_PIN_COUNT + 1]]

                    except Exception as fault:
                        log.warning(f"[SIMULATED] Invalid TARGETS payload: {command_string} - {fault}!")

                    else:
                        for offset, volts in enumerate(raw_values, start=1):

                            pwm_value = self.voltage_to_pulse(volts)
                            self.update_pin_values(offset, pwm_value)

                        log.info(f"[SIMULATED] TARGETS applied: {raw_values}")

                else:
                    log.info(f"[SIMULATED] Received command: {command_string}")
                return

            if not self.serial or not self.serial.is_open:
                raise ConnectionError("ERROR: Serial port not open!")

            self.serial.write((command_string + "\n").encode("utf-8"))
            log.info(f"Command sent to board: {command_string}...")

        except Exception as fault:
            log.error(f"ERROR: Failed to send command '{command_string}' - {fault}!")

    def get_data(self) -> pd.DataFrame:
        """Return a copy of the current rolling DataFrame."""
        with self.data_lock:
            data_frame = self.data_frame.copy()
            return data_frame

    def get_status(self) -> str:
        """Return a human-readable connection/status string."""
        if self.offline:
            return "Simulating operation..."
        if self.serial and getattr(self.serial, "is_open", False):
            return f"Connected via {self.port}..."
        return "Connecting..."

    def get_pin_value(self, i: int, value: int):
        """Clamp a requested pin value based on pin type (PWM pins 1..5 vs switch pin 6)."""

        if not (1 <= i <= 6):
            log.warning("WARNING: Pin index must be 1-6!")
            return None
        
        if i <= 5:
            return max(0, min(MAX_MODULATION_VALUE, int(value)))
        return 1 if int(value) else 0

    def set_pin_voltage(self, i: int, value: int):
        """Set a single pin by index (1..6) using raw PWM duty (pins 1..5) or 0/1 (pin 6)."""
        if i < 1 or i > 6:
            raise ValueError("ERROR: index must be 1-6!")
        
        if i <= 5:
            value = max(0, min(MAX_MODULATION_VALUE, int(value)))
        else:
            value = 1 if int(value) else 0

        self.send_command(f"PIN {int(i)} {int(value)}")

    def set_pin_voltages(self, voltages):
        """Set the first 5 control pins using volt targets (sends a TARGETS command).

        Args:
            voltages: Iterable of voltages; first 5 values are used.
        """

        if voltages is None:
            raise ValueError("ERROR: Missing voltage targets!")

        try:
            values = [float(voltage_output) for voltage_output in voltages]
        except Exception as fault:
            raise ValueError(f"ERROR: Invalid voltage target - {fault}!")

        if len(values) < CONTROL_PIN_COUNT:
            raise ValueError(f"ERROR: Invalid number of outputs requested, I need {CONTROL_PIN_COUNT} voltages!")

        target_voltages = [self.clamp_voltage(voltage_output) for voltage_output in values[:CONTROL_PIN_COUNT]]
        voltage_payload = "TARGETS " + " ".join(f"{voltage_output:.6f}" for voltage_output in target_voltages)
        self.send_command(voltage_payload)

    def set_pwm(self, channel: int, duty: int):
        """Alias for setting PWM duty on control pins 1..5."""

        if channel < 1 or channel > 5:
            raise ValueError("ERROR: Pulse source must be a channel in range: 1-5!")
        
        self.set_pin_voltage(channel, duty)

    def set_switch_timing(self, timing_input: float):
        """Set the switch timing (microseconds) used by the embedded controller."""
        try:
            switch_timing = float(timing_input)

        except Exception as fault:
            log.warning(f"WARNING: Switch timing not set: {fault}!")
            return

        board_switch_timing = max(1.0, min(20.0, switch_timing))
        self.switch_timing = board_switch_timing

        if self.offline:
            try:
                log.info(f"[SIMULATED] Switch timing set to {board_switch_timing:.1f} us")

            except Exception as fault:
                log.warning(f"[SIMULATED] WARNING: Failed to log switch timing: {fault}!")
                pass
            return

        try:
            self.send_command(f"SWITCH_PERIOD {int(board_switch_timing)}")
        except Exception as fault:
            log.error(f"ERROR: Failed to send switch timing '{board_switch_timing}': {fault}!")

    def set_pin_by_name(self, name: str, value: int):
        """Set a pin name."""

        i = self._name_to_pin_index(name)

        if not (1 <= i <= 6):
            raise ValueError("ERROR: Unknown pin index!")
        
        if i <= 5:
            value = max(0, min(MAX_MODULATION_VALUE, int(value)))

        else:
            value = 1 if int(value) else 0
        self.send_command(f"PIN {int(i)} {int(value)}")

    def get_pins(self):
        """Return cached pin names/values and the last-update timestamp."""

        names = [
            "squeeze_plate",
            "ion_source",
            "wein_filter",
            "cone_1",
            "cone_2",
            "switch_logic",
        ]

        return {"names": names, "values": list(self.pins), "timestamp": self.pins_timestamp,}

    def stop(self):
        """Stop background threads and close the serial connection."""

        log.info("Backend thread disconnecting...")
        self.alive.clear()
        self.disconnect()
        log.info("Backend thread disconnected...")

    # ------------------------------------------------------------------ #
    #                         Training sweep control                     #  
    # ------------------------------------------------------------------ #

    def _RNN_training_sweeps(self, minimum_voltage: float, maximum_voltage: float, step: float, hold_time: float, epochs: int, baseline_levels: int | None = None, factorial_levels: int | None = None, random_samples: int | None = None):
        """Internal worker for parameter sweeps + optional RNN training."""

        try:

            self.sweep_status = {"state": "running", "progress": 0.0, "message": ""}

            try:
                minimum_voltage = float(minimum_voltage)
                maximum_voltage = float(maximum_voltage)

            except Exception as fault:
                log.warning(f"WARNING: Invalid voltage range - {fault}! Reverting to defaults...")
                minimum_voltage, maximum_voltage = 0.0, 3.3

            if minimum_voltage > maximum_voltage:
                log.warning("WARNING: Minimum voltage greater than maximum voltage! Swapping values...")
                minimum_voltage, maximum_voltage = maximum_voltage, minimum_voltage

            span = max(0.0, maximum_voltage - minimum_voltage)

            try:
                step = float(step)

            except Exception as fault:
                log.warning(f"WARNING: Invalid step value - {fault}! Reverting to defaults...")
                step = 0.05

            if step <= 0:
                step = max(0.01, span / 25.0) if span > 0 else 0.05

            try:
                hold_time = max(0.01, float(hold_time))

            except Exception as fault:
                log.warning(f"WARNING: Invalid hold time - {fault}! Reverting to defaults...")
                hold_time = 0.05

            grid = np.arange(minimum_voltage, maximum_voltage + 1e-9, step, dtype=float)

            if grid.size == 0:
                log.warning("WARNING: Grid size must be  non-zero! Using all available information...")
                grid = np.array([minimum_voltage], dtype=float)

            grid = np.clip(grid, minimum_voltage, maximum_voltage)

            try:
                baseline_count = int(baseline_levels) if baseline_levels is not None else (3 if span > 0 else 1)

            except Exception as fault:
                log.warning(f"WARNING: Invalid baseline levels - {fault}! Reverting to defaults...")
                baseline_count = 3 if span > 0 else 1
                
            baseline_count = max(1, baseline_count)
            baseline_levels = np.linspace(minimum_voltage, maximum_voltage, num=baseline_count, dtype=float)
            baseline_levels = np.unique(np.round(baseline_levels, 6))

            if baseline_levels.size == 0:
                log.warning("WARNING: Baseline levels size must be non-zero! Using minimum voltage...")
                baseline_levels = np.array([minimum_voltage], dtype=float)

            try:
                factorial_level_count = int(factorial_levels) if factorial_levels is not None else (3 if span > 0 else 1)

            except Exception as fault:
                log.warning(f"WARNING: Invalid factorial levels - {fault}! Reverting to defaults...")
                factorial_level_count = 3 if span > 0 else 1

            factorial_level_count = max(1, factorial_level_count)
            factorial_levels = np.linspace(minimum_voltage, maximum_voltage, num=factorial_level_count, dtype=float)
            factorial_levels = np.clip(factorial_levels, minimum_voltage, maximum_voltage)

            if random_samples is None:
                log.warning("WARNING: Random samples not specified! Using default value...")
                random_sample_count = max(int(len(grid) * CONTROL_PIN_COUNT), 20)

            else:
                try:
                    random_sample_count = int(random_samples)

                except Exception as fault:
                    log.warning(f"WARNING: Invalid random sample - {fault}! Reverting to defaults...")
                    random_sample_count = 0

                if random_sample_count <= 0:
                    log.warning("WARNING: Random sample count must be positive! Using default value...")
                    random_sample_count = max(int(len(grid) * CONTROL_PIN_COUNT), 20)

            stage1_steps = int(len(baseline_levels) * CONTROL_PIN_COUNT * len(grid))
            stage2a_steps = int(len(factorial_levels) ** CONTROL_PIN_COUNT)

            total_steps = stage1_steps + stage2a_steps + random_sample_count

            if total_steps <= 0:
                log.warning("WARNING: Total number of sweep steps must be positive! Using 1...")
                total_steps = 1

            random_number_generator = np.random.default_rng()
            completion_indicator = 0

            class SweepAbort(Exception):
                log.warning("WARNING: Sweep aborted by user!")
                pass

            def run_step(target_vector, stage_label):
                nonlocal completion_indicator

                if self.sweep_cancel.is_set():
                    raise SweepAbort

                try:
                    self.set_pin_voltages(list(target_vector))
                except Exception as fault:
                    log.warning(f"ERROR: Failed to set sweep voltages - {fault}!")

                time.sleep(hold_time)
                completed += 1

                self.sweep_status["progress"] = min(1.0, completed / total_steps)
                self.sweep_status["message"] = stage_label

            try:
                for baseline in baseline_levels:

                    base_vector = [baseline] * CONTROL_PIN_COUNT

                    for pin_index in range(CONTROL_PIN_COUNT):
                        for value in grid:

                            targets = list(base_vector)
                            targets[pin_index] = float(value)

                            run_step(targets,f"Sweep 1: pin {pin_index + 1} sweep @ baseline {baseline:.2f} V",)

                combo_total = max(1, stage2a_steps)
                combo_index = 0

                for combination_sweep_index in itertools.product(factorial_levels, repeat=CONTROL_PIN_COUNT):

                    combo_index += 1
                    run_step(combination_sweep_index, f"Sweep 2: factorial sweep for the RNN ({combo_index}/{combo_total})")

                for sample_index in range(1, random_sample_count + 1):

                    random_targets = random_number_generator.uniform(minimum_voltage, maximum_voltage, CONTROL_PIN_COUNT)
                    run_step(random_targets.tolist(), f"Stage 3: random sampling for the RNN ({sample_index}/{random_sample_count})",)

            except SweepAbort:
                self.sweep_status.update({"state": "aborted", "message": "Sweep aborted by user..."})
                return

            data_frame = self.get_data()

            try:
                if data_frame.empty:

                    self.sweep_status.update({
                        "state": "failed",
                        "message": "ERROR: Sweep Failed! No data collected during sweep attempt...",
                        "progress": 0.0,
                    })

                    return

                if self.save_dataset_enabled:
                    try:

                        ordered_columns = [
                            "timestamp",
                            "voltage",
                            "pin_1",
                            "pin_2",
                            "pin_3",
                            "pin_4",
                            "pin_5",
                            "switch_logic",
                            "raw_message",
                        ]

                        available_columns = [column for column in ordered_columns if column in data_frame.columns]

                        dataset = data_frame.loc[:, available_columns].rename(
                            columns={
                                "pin_1": "squeeze_plate",
                                "pin_2": "ion_source",
                                "pin_3": "wein_filter",
                                "pin_4": "cone_1",
                                "pin_5": "cone_2",
                            }
                        )

                        time_stamp = time.strftime("%Y%m%d_%H%M%S")
                        file_path = f"sweep_dataset_{time_stamp}.csv"
                        dataset.to_csv(file_path, index=False)
                        log.info(f"Sweep dataset saved to {file_path}")

                    except Exception as fault:
                        log.error(f"ERROR: Failed to save sweep dataset - {fault}!")

                if train_model is not None:
                    try:
                        metrics = train_model(data_frame, number_of_epochs=int(max(1, epochs)))
                        current_time = time.time()

                        self.last_training_time_stamp = current_time
                        self.last_update_time_stamp = current_time

                        try:
                            if isinstance(metrics, dict):

                                ML_METRICS.record_training_update(
                                    current_time,
                                    loss = metrics.get("loss"),
                                    r2 = metrics.get("r2"),
                                    source = "sweep",
                                )

                        except Exception as fault:
                            log.warning(f"WARNING: Failed to record sweep training metrics - {fault}!")
                            pass

                        self.sweep_status.update({
                            "state": "completed",
                            "message": f"Sweep & RNN Training complete! {len(data_frame)} samples collected over {int(max(1, epochs))} epochs",
                            "progress": 1.0,
                        })

                    except Exception as fault:

                        self.sweep_status.update({
                            "state": "failed",
                            "message": f"ERROR: Training failed after sweep - {fault}!",
                        })

                else:
                    self.sweep_status.update({
                        "state": "completed",
                        "message": f"Data sweep complete (without training) - {len(data_frame)} samples collected!",
                        "progress": 1.0,
                    })

            except Exception as fault:
                self.sweep_status.update({
                    "state": "failed",
                    "message": f"ERROR: Sweep post-processing error - {fault}!",
                })

        except Exception as fault:
            self.sweep_status.update({
                "state": "failed",
                "message": f"ERROR: General sweep error - {fault}!",
            })

    def start_training_sweep(
        self,
        min_voltage: float = 0.0,
        max_voltage: float = 3.3,
        step: float = 0.05,
        step_linger_time: float = 0.05,
        epochs: int = 10,
        baseline_levels: int | None = None,
        factorial_levels: int | None = None,
        random_samples: int | None = None,
    ):
        """Start a background training sweep.

        Returns:
            True if the sweep was started, False if one is already running.
        """

        if self.sweep_thread and self.sweep_thread.is_alive():
            log.info("Sweep already running...")
            return False
        
        self.sweep_cancel.clear()
        self.sweep_status = {"state": "queued", "progress": 0.0, "message": ""}
        self.sweep_thread = threading.Thread(target=self._RNN_training_sweeps,args=(
            min_voltage, 
            max_voltage, 
            step, 
            step_linger_time, 
            epochs, 
            baseline_levels, 
            factorial_levels, 
            random_samples
            ),
            daemon=True,
        )

        self.sweep_thread.start()

        return True

    def get_sweep_status(self) -> dict:
        """Return the latest sweep status (state/progress/message)."""

        return dict(self.sweep_status)

    def stop_training_sweep(self):
        """Request cancellation of an in-progress sweep."""

        self.sweep_cancel.set()


# ------------------------------------------------------------------------- #
#                               Board Access                                #    
# ------------------------------------------------------------------------- #

# change once connection is ensured to prevent error logs during startup
OFFLINE = False
status = OFFLINE

Back_End_Controller = SerialBackend(status = status)
Data_Reciever = Back_End_Controller

get_data = Back_End_Controller.get_data
send_command = Back_End_Controller.send_command

set_pin_voltage = Back_End_Controller.set_pin_voltage
set_pin_voltages = Back_End_Controller.set_pin_voltages
set_pwm = Back_End_Controller.set_pwm
set_switch = Back_End_Controller.set_switch_timing
set_switch_timing = getattr(Back_End_Controller, "set_switch_timing", None)
set_pin_by_name = Back_End_Controller.set_pin_by_name
get_pins = Back_End_Controller.get_pins
set_window_update_time = getattr(Back_End_Controller, "set_window_update_time", None)
set_online_learning_rate = getattr(Back_End_Controller, "set_online_learning_rate", None)
get_online_update_config = getattr(Back_End_Controller, "get_online_update_config", None)

lines = Back_End_Controller.lines
get_status = Back_End_Controller.get_status

# ------------------------------------------------------------------------- #
#                           Model Status Livestream                         #
# ------------------------------------------------------------------------- #

def get_model_info() -> dict:
    """Returns a lightweight snapshot of model/online-update state to be displayed opn the UI Dashboard."""

    status_snapshot = {
        "last_train_ago_sec": None,
        "last_online_update_ago_sec": None,
        "online_window_seconds": None,
        "online_updates_enabled": None,
        "learning_rate": None,
        "sweep_state": None,
    }

    current_time = time.time()

    try:
        ts = getattr(Back_End_Controller, "last_training_time_stamp", None)

        if ts is not None:
            status_snapshot["last_train_ago_sec"] = max(0.0, float(current_time - float(ts)))

    except Exception as fault:
        log.warning(f"WARNING: Couldnot access the last training time step: {fault}!")
        pass
    
    try:
        online_ts = getattr(Back_End_Controller, "last_update_time_stamp", None)

        if online_ts is not None:
            status_snapshot["last_online_update_ago_sec"] = max(0.0, float(current_time - float(online_ts)))
   
    except Exception as fault:
        log.warning(f"WARNING: Couldnot access the last online update time step: {fault}!")
        pass

    try:
        status_snapshot["online_window_seconds"] = float(getattr(Back_End_Controller, "online_window_seconds", None))
    
    except Exception as fault:
        log.warning(f"WARNING: Couldnot access the online window seconds: {fault}!")
        status_snapshot["online_window_seconds"] = None

    try:
        status_snapshot["online_updates_enabled"] = bool(getattr(Back_End_Controller, "online_update_enabled", None))
    
    except Exception as fault:
        log.warning(f"WARNING: Couldnot access the online updates enabled status: {fault}!")
        status_snapshot["online_updates_enabled"] = None
    
    try:
        status_snapshot["learning_rate"] = _get_rnn_learning_rate()
    
    except Exception as fault:
        log.warning(f"WARNING: Couldnot access the learning rate: {fault}!")
        status_snapshot["learning_rate"] = None
   
    try:
        status_snapshot["momentum"] = _get_model_momentum()
   
    except Exception as fault:
        log.warning(f"WARNING: Couldnot access the momentum: {fault}!")
        status_snapshot["momentum"] = None
   
    try:
        status_snapshot["sweep_state"] = str(getattr(Back_End_Controller, "sweep_status", {}).get("state", "idle"))
    
    except Exception as fault:
        log.warning(f"WARNING: Couldnot access the sweep state: {fault}!")
        status_snapshot["sweep_state"] = None
    
    return status_snapshot


# -------------------------------------------------------------------------
#                   Machine Learning Metrics Interface
# -------------------------------------------------------------------------

ML_METRICS = MetricCollector(maxlen=4000)

def get_ml_metrics():
    """Update and return the ML metrics for display on the dashboard."""

    try:
        latest_reading = Back_End_Controller.get_latest_point()
        pin_values = Back_End_Controller.get_pins().get("values", [])

        if latest_reading is not None:
            timestamps, voltages = latest_reading
            ML_METRICS.update_stream_from_backend(timestamps, voltages, pin_values)

    except Exception as fault:
        log.warning(f"WARNING: Could not update ML metrics from backend: {fault}!")
        pass
    return ML_METRICS.dashboard_snapshot()


def push_ml_features(features, names=None):
    """Record a set of feature values for debugging."""

    try:
        ML_METRICS.record_features(features, feature_names = names)
    except Exception as fault:
        log.warning(f"WARNING: Could not record ML features: {fault}!")
        pass


def push_ml_saliency(saliency):
    """Record feature saliency values for debugging."""

    try:
        ML_METRICS.record_feature_saliencies(saliency)
    except Exception as fault:
        log.warning(f"WARNING: Could not record ML saliencies: {fault}!")
        pass


def save_model_parameters():
    """Saves the model parameters via user input in the backend controller."""

    return Back_End_Controller.save_model_parameters()


def compute_feature_importance(max_samples: int = 200, num_permutations: int = 20):
    """Computes feature saliency from backend data."""
    
    return Back_End_Controller.compute_feature_importance(max_samples=max_samples, num_permutations=num_permutations)

#-------------------------------------------------------------------------#
#                   Training Sweep Controls
#-------------------------------------------------------------------------#

start_training_sweep = getattr(Back_End_Controller, "start_training_sweep", None)
get_sweep_status = getattr(Back_End_Controller, "get_sweep_status", None)
stop_training_sweep = getattr(Back_End_Controller, "stop_training_sweep", None)


# -------------------------------------------------------------------------
#                             Manual testing
# -------------------------------------------------------------------------

if __name__ == "__main__":

    log.info("Starting backend in standalone mode...")

    try:
        while True:
            time.sleep(1)
            data_frame = Back_End_Controller.get_data()
            
            if not data_frame.empty:
                log.info(f"Latest: {data_frame['voltage'].iloc[-1]:.3f} V ({len(data_frame)} samples)")

    except KeyboardInterrupt:
        Back_End_Controller.stop()
        log.info("Backend stopped...")
