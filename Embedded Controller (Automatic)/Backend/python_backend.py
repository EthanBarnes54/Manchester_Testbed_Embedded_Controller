# --------- Backend Communication Layer for the ESP32 Cntroller --------- #

# Handles serial communication between the ESP32 and the Python environment.
# Provides live data streaming, thread-safe buffering, and command
# transmission. Designed for use with the data loggier, dashboard, signal
# pipeline, and RNN controller modules.

#------------------------------------------------------------------------#

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
    from python_RNN_Controller import train_model
except Exception:
    train_model = None


# -------------------------------------------------------------------------
#                             Logging setup
# -------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ESP32_Backend")


# -------------------------------------------------------------------------
#                             Configuration
# -------------------------------------------------------------------------

SERIAL_PORT = "COM4"  # Adjust for OTA
BAUD_RATE = 115200
RETRY_DELAY = 3.0
MAX_QUEUE_SIZE = 2000

OFFLINE = os.getenv("OFFLINE", "").strip() not in ("", "0", "false", "False")


# -------------------------------------------------------------------------
#                             Backend Class
# -------------------------------------------------------------------------


class SerialBackend:

    def __init__(self, port=SERIAL_PORT, baud=BAUD_RATE, offline: bool = False):
        self.port = port
        self.baud = baud
        self.serial = None

        self.alive = threading.Event()
        self.lines = queue.Queue(maxsize=MAX_QUEUE_SIZE)
        self.data_lock = threading.Lock()
        self.data_frame = pd.DataFrame(columns=["timestamp", "dac", "voltage", "raw_message"])

        self.offline = offline
        self.target_voltage = 1.0
        self.live_voltage = 1.0

        self.pins = [0, 0, 0, 0, 0, 0]
        self.pins_timestamp = 0.0

        self.sweep_thread = None
        self.sweep_status = {"state": "idle", "progress": 0.0, "message": ""}
        self.sweep_cancel = threading.Event()

        # Optional dataset persistence after sweep (configurable from dashboard)
        self.save_dataset_enabled = False

        # Timestamp of last successful model training (seconds since epoch)
        self.last_training_ts = None

        self.thread = threading.Thread(target=self.run, daemon=True)
        self.alive.set()
        self.thread.start()

    # ------------------------------------------------------------------
    #                    Connect / Disconnect Switches
    # ------------------------------------------------------------------

    def connect(self):
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
        if self.serial and self.serial.is_open:
            try:
                log.info("Serial closing...")
                self.serial.close()
                log.info("Serial now closed...")

            except Exception:
                pass

        self.serial = None

    def set_save_dataset_enabled(self, enabled: bool):
        try:
            self.save_dataset_enabled = bool(enabled)
        except Exception:
            self.save_dataset_enabled = False

    def get_latest_point(self):
        with self.data_lock:
            if self.data_frame.empty:
                return None
            row = self.data_frame.iloc[-1]
            return float(row["timestamp"]), float(row["voltage"])

    @staticmethod
    def _name_to_index(pin_name: str) -> int:
        if pin_name is None:
            return 0
        
        pin_number = str(pin_name).strip().lower()

        mapping = {
            "squeeze_plate": 1,
            "ion_source": 2,
            "wein_filter": 3,
            "cone_1": 4,
            "cone_2": 5,
            "switch_logic": 6,
        }

        if pin_number in mapping:
            return mapping[pin_number]
        try:
            i = int(pin_number)
            return i if 1 <= i <= 6 else 0
        except Exception:
            return 0

    def update_pin_values(self, pin_index: int, pin_value: int) -> bool:

        if not (1 <= int(pin_index) <= 6):
            return False
        
        clamped_pin_value = self.get_pin_value(int(pin_index), int(pin_value))

        if clamped_pin_value is None:
            return False
        
        self.pins[int(pin_index) - 1] = clamped_pin_value
        self.pins_timestamp = time.time()

        return True

    def run(self):

        while self.alive.is_set():
            if self.offline:

                #Change once rig has been wired and finalised 
                timestamp = time.time()
                noise = (random.random() - 0.5) * 0.5
                v = max(0.0, min(3.3, self.target_voltage + noise))
                message = f"MEASURED {v:.3f}"

                if not self.lines.full():
                    self.lines.put((timestamp, message))

                with self.data_lock:

                    self.data_frame.loc[len(self.data_frame)] = {
                        "timestamp": timestamp,
                        "dac": float(self.live_voltage),
                        "voltage": v,
                        "raw_message": message,
                    }

                    self.data_frame = self.data_frame.tail(1000).reset_index(drop=True)

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
                        voltage = None

                    with self.data_lock:
                        self.data_frame.loc[len(self.data_frame)] = {
                            "timestamp": timestamp,
                            "dac": float(self.live_voltage),
                            "voltage": voltage,
                            "raw_message": message,
                        }

                        self.data_frame = self.data_frame.tail(1000).reset_index(drop=True)

                elif message.startswith("PINS"):
                    try:
                        for pin_assignment in message.split()[1:]:

                            if "=" not in pin_assignment:
                                continue

                            name, value_string = pin_assignment.split("=", 1)

                            try:
                                pin_value = int(float(value_string))
                            except ValueError:
                                continue

                            pin_index = self._name_to_index(name)
                            self.update_pin_values(pin_index, pin_value)
                    except Exception:
                        pass
                elif message.startswith("ACK PIN"):
                    try:
                        tokens = message.split()
                        self.update_pin_values(int(tokens[2]), int(tokens[3]))
                    except Exception:
                        pass

            except serial.SerialException as serial_fault:
                log.warning(f"ERROR: Serial exception - {serial_fault}! Reconnecting...")
                self.disconnect()
                time.sleep(RETRY_DELAY)

            except Exception as fault:
                log.error(f"ERROR: Unexpected - {fault}!")
                time.sleep(0.5)

    # ------------------------------------------------------------------
    #                               Methods
    # ------------------------------------------------------------------

    def send_command(self, command_string: str):
        try:
            if self.offline:

                pin_assignments = command_string.strip().split()

                if len(pin_assignments) == 2 and pin_assignments[0].upper() == "SET":
                    try:
                        v = float(pin_assignments[1])
                        v_target = max(0.0, min(3.3, v))
                        self.target_voltage = v_target
                        self.live_voltage = v_target

                        log.info(f"[SIM] Target voltage set to {self.target_voltage:.3f} V")

                    except ValueError:
                        log.warning(f"[SIM] ERROR: Invalid SET value in command - {command_string}!")

                elif len(pin_assignments) == 3 and pin_assignments[0].upper() == "PIN":
                    try:
                        idx = self._name_to_index(pin_assignments[1])
                        val = int(float(pin_assignments[2]))

                        if self.update_pin_values(idx, val):
                            log.info(f"[SIM] PIN {idx} set to {self.pins[idx-1]}")
                        else:
                            log.warning(f"[SIM] PIN index out of range: {idx}")
                    except Exception:
                        log.warning(f"[SIM] Invalid PIN command: {command_string}")

                else:
                    log.info(f"[SIM] Received command: {command_string}")
                return

            if not self.serial or not self.serial.is_open:
                raise ConnectionError("ERROR: Serial port not open!")

            self.serial.write((command_string + "\n").encode("utf-8"))
            log.info(f"Command sent to board: {command_string}...")

            pin_assignments = command_string.strip().split()

            if len(pin_assignments) == 2 and pin_assignments[0].upper() == "SET":
                try:
                    voltage = float(pin_assignments[1])
                    self.live_voltage = max(0.0, min(3.3, voltage))
                except ValueError:
                    pass
        except Exception as fault:
            log.error(f"ERROR: Failed to send command '{command_string}' - {fault}!")

    def get_data(self) -> pd.DataFrame:
        with self.data_lock:
            return self.data_frame.copy()

    def get_status(self) -> str:
        if self.offline:
            return "Simulated"
        if self.serial and getattr(self.serial, "is_open", False):
            return f"Connected via {self.port}..."
        return "Connecting..."

    def get_pin_value(self, i: int, value: int):
        if not (1 <= i <= 6):
            return None
        if i <= 5:
            return max(0, min(1023, int(value)))
        return 1 if int(value) else 0

    def set_pin_voltage(self, i: int, value: int):
        if i < 1 or i > 6:
            raise ValueError("ERROR: index must be 1-6!")
        if i <= 5:
            value = max(0, min(1023, int(value)))
        else:
            value = 1 if int(value) else 0
        self.send_command(f"PIN {int(i)} {int(value)}")

    def set_pwm(self, channel: int, duty: int):
        if channel < 1 or channel > 5:
            raise ValueError("ERROR: PWM channel must be 1-5!")
        self.set_pin_voltage(channel, duty)

    # REMOVE ONCE RIG IS FINALISED
    def set_switch(self, on: bool):
        self.set_pin_voltage(6, 1 if on else 0)
    def set_pin(self, i: int, value: int):
        self.set_pin_voltage(i, value)

    def set_pin_by_name(self, name: str, value: int):
        i = self._name_to_index(name)
        if not (1 <= i <= 6):
            raise ValueError("ERROR: Unknown pin name!")
        if i <= 5:
            value = max(0, min(1023, int(value)))
        else:
            value = 1 if int(value) else 0
        self.send_command(f"PIN {int(i)} {int(value)}")

    def get_pins(self):
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
        log.info("Backend thread stopping...")
        self.alive.clear()
        self.disconnect()
        log.info("Backend thread stopped...")

    # ------------------------------------------------------------------
    #                         Training sweep control
    # ------------------------------------------------------------------

    #CHANGE ONCE RIG IS FINALISED - HENCE WHY EPOCHS PARAMETER IS NOT USED

    def _sweep_worker(self, minimum_voltage: float, maximum_voltage: float, step: float, hold_time: float, epochs: int):
        try:
            self.sweep_status = {
                "state": "running",
                "progress": 0.0,
                "message": "",
            }

            grid = np.arange(minimum_voltage, maximum_voltage + 1e-9, step, dtype=float)
            total = len(grid)

            for i, v in enumerate(grid):

                if self.sweep_cancel.is_set():
                    self.sweep_status.update({"state": "aborted", "message": "Sweep aborted by user..."})
                    return
                
                self.send_command(f"SET {v:.3f}")
                time.sleep(hold_time)
                self.sweep_status["progress"] = (i + 1) / max(total, 1)

            if self.sweep_cancel.is_set():
                self.sweep_status.update({"state": "aborted", "message": "Sweep aborted before training..."})
                return

            data_frame = self.get_data()

            try:
                if data_frame.empty:
                    self.sweep_status.update({
                        "state": "failed",
                        "message": "No data collected during sweep.",
                        "progress": 0.0,
                    })
                    return

                if "dac" not in data_frame.columns and "voltage" in data_frame.columns:
                    data_frame = data_frame.assign(dac=self.live_voltage)

                if self.save_dataset_enabled:
                    try:
                        ts = time.strftime("%Y%m%d_%H%M%S")
                        path = f"sweep_dataset_{ts}.csv"
                        data_frame.to_csv(path, index=False)
                        log.info(f"Sweep dataset saved to {path}")
                    except Exception as save_fault:
                        log.warning(f"Failed to save sweep dataset: {save_fault}")

                if train_model is not None:
                    try:
                        train_model(data_frame, number_of_epochs=int(max(1, epochs)))
                        self.last_training_ts = time.time()
                        self.sweep_status.update({
                            "state": "completed",
                            "message": f"Sweep + training complete. {len(data_frame)} samples; epochs={int(max(1, epochs))}",
                            "progress": 1.0,
                        })
                    except Exception as train_fault:
                        self.sweep_status.update({
                            "state": "failed",
                            "message": f"Training failed after sweep: {train_fault}",
                        })
                else:
                    self.sweep_status.update({
                        "state": "completed",
                        "message": f"Data sweep complete (training unavailable). {len(data_frame)} samples.",
                        "progress": 1.0,
                    })

            except Exception as fault:
                self.sweep_status.update({
                    "state": "failed",
                    "message": f"ERROR: Sweep post-processing error - {fault}!",
                })

        except Exception as fault:
            # Catch any error from the overall sweep worker
            self.sweep_status.update({
                "state": "failed",
                "message": f"ERROR: Sweep error - {fault}!",
            })

    #CHANGE ONCE RIG IS FINALISED - SWEEP OVER ENTIRE VOLTAGE RANGES FOR ALL PINS
    def start_training_sweep(self, min_v: float = 0.0, max_v: float = 3.3, step: float = 0.05, dwell_s: float = 0.05, epochs: int = 10,):

        if self.sweep_thread and self.sweep_thread.is_alive():
            log.info("Sweep already running; ignoring start request.")
            return False
        
        self.sweep_cancel.clear()
        self.sweep_status = {"state": "queued", "progress": 0.0, "message": ""}
        self.sweep_thread = threading.Thread(target=self._sweep_worker, args=(min_v, max_v, step, dwell_s, epochs), daemon=True,)
        self.sweep_thread.start()

        return True

    def get_sweep_status(self) -> dict:
        return dict(self.sweep_status)

    def stop_training_sweep(self):
        self.sweep_cancel.set()


# -------------------------------------------------------------------------
#                   Global event for board access
# -------------------------------------------------------------------------

Back_End_Controller = SerialBackend(offline=OFFLINE)
reader = Back_End_Controller

get_data = Back_End_Controller.get_data
send_command = Back_End_Controller.send_command

set_pin_voltage = Back_End_Controller.set_pin_voltage
set_pwm = Back_End_Controller.set_pwm
set_switch = Back_End_Controller.set_switch
set_pin = Back_End_Controller.set_pin
set_pin_by_name = Back_End_Controller.set_pin_by_name
get_pins = Back_End_Controller.get_pins

lines = Back_End_Controller.lines
get_status = Back_End_Controller.get_status

# -------------------------------------------------------------------------
#                   Model info (for dashboard)
# -------------------------------------------------------------------------

def get_model_info() -> dict:
    try:
        ts = getattr(Back_End_Controller, "last_training_ts", None)
    except Exception:
        ts = None
    if ts is None:
        return {"last_train_ago_sec": None}
    try:
        now = time.time()
        return {"last_train_ago_sec": max(0.0, float(now - float(ts)))}
    except Exception:
        return {"last_train_ago_sec": None}


# -------------------------------------------------------------------------
#                   Machine Learning Metrics Interface
# -------------------------------------------------------------------------

ML_METRICS = MetricCollector(maxlen=4000)

def get_ml_metrics():

    try:
        latest_reading = Back_End_Controller.get_latest_point()
        pin_values = Back_End_Controller.get_pins().get("values", [])

        if latest_reading is not None:
            timestamps, voltages = latest_reading
            ML_METRICS.update_stream_from_backend(timestamps, voltages, pin_values)

    except Exception:
        pass
    return ML_METRICS.dashboard_snapshot()


def push_ml_features(features, names=None):
    try:
        ML_METRICS.record_features(features, feature_names = names)
    except Exception:
        pass


def push_ml_saliency(saliency):
    try:
        ML_METRICS.record_feature_saliencies(saliency)
    except Exception:
        pass

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
