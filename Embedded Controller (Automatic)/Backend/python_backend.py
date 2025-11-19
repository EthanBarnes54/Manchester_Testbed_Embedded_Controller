# --------- Backend Communication Layer for the ESP32 Cntroller --------- #

# Handles serial communication between the ESP32 and the Python environment.
# Provides live data streaming, thread-safe buffering, and command
# transmission. Designed for use with the data loggier, dashboard, signal
# pipeline, and RNN controller modules.

#------------------------------------------------------------------------#

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

MAX_MODULATION_VALUE = 1023
MAX_CONTROL_VOLTAGE = 3.3
CONTROL_PIN_COUNT = 5


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

        self.offline = offline

        self.pins = [0, 0, 0, 0, 0, 0]
        self.pins_timestamp = 0.0
        self.switch_timing_us = None

        self.sweep_thread = None
        self.sweep_status = {"state": "idle", "progress": 0.0, "message": ""}
        self.sweep_cancel = threading.Event()

        self.save_dataset_enabled = False
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

    @staticmethod
    def clamp_pwm_value(value) -> int:
        try:
            v = int(round(float(value)))
        except Exception:
            return 0
        return max(0, min(MAX_MODULATION_VALUE, v))

    @staticmethod
    def clamp_voltage(voltage: float) -> float:
        try:
            v = float(voltage)
        except Exception:
            return 0.0
        return max(0.0, min(MAX_CONTROL_VOLTAGE, v))

    @classmethod
    def pwm_to_voltage(cls, value: float) -> float:
        pwm = cls.clamp_pwm_value(value)
        return (pwm / MAX_MODULATION_VALUE) * MAX_CONTROL_VOLTAGE

    @classmethod
    def voltage_to_pwm(cls, voltage: float) -> int:
        volts = cls.clamp_voltage(voltage)
        scale = (volts / MAX_CONTROL_VOLTAGE) * MAX_MODULATION_VALUE
        return cls.clamp_pwm_value(scale)

    def update_pin_values(self, pin_index: int, pin_value: int) -> bool:

        if not (1 <= int(pin_index) <= 6):
            return False
        
        clamped_pin_value = self.get_pin_value(int(pin_index), int(pin_value))

        if clamped_pin_value is None:
            return False
        
        self.pins[int(pin_index) - 1] = clamped_pin_value
        self.pins_timestamp = time.time()

        return True

    def _append_measurement(self, timestamp: float, voltage: float, message: str):
        snapshot = list(self.pins)
        row = {
            "timestamp": float(timestamp),
            "voltage": None if voltage is None else float(voltage),
            "pin_1": snapshot[0],
            "pin_2": snapshot[1],
            "pin_3": snapshot[2],
            "pin_4": snapshot[3],
            "pin_5": snapshot[4],
            "switch_logic": snapshot[5],
            "raw_message": message,
        }
        with self.data_lock:
            self.data_frame.loc[len(self.data_frame)] = row
            self.data_frame = self.data_frame.tail(1000).reset_index(drop=True)

    def run(self):

        while self.alive.is_set():
            if self.offline:

                #Change once rig has been wired and finalised 
                timestamp = time.time()
                analog_snapshot = [self.pwm_to_voltage(p) for p in self.pins[:CONTROL_PIN_COUNT]]
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
                if not pin_assignments:
                    return

                cmd = pin_assignments[0].upper()

                if cmd == "PIN" and len(pin_assignments) == 3:
                    try:
                        idx = self._name_to_index(pin_assignments[1])
                        val = self.clamp_pwm_value(pin_assignments[2])

                        if self.update_pin_values(idx, val):
                            log.info(f"[SIM] PIN {idx} set to {self.pins[idx-1]}")
                        else:
                            log.warning(f"[SIM] PIN index out of range: {idx}")
                    except Exception:
                        log.warning(f"[SIM] Invalid PIN command: {command_string}")

                elif cmd == "TARGETS" and len(pin_assignments) >= CONTROL_PIN_COUNT + 1:
                    try:
                        raw_values = [float(token) for token in pin_assignments[1 : CONTROL_PIN_COUNT + 1]]
                    except Exception:
                        log.warning(f"[SIM] Invalid TARGETS payload: {command_string}")
                    else:
                        for offset, volts in enumerate(raw_values, start=1):
                            pwm_value = self.voltage_to_pwm(volts)
                            self.update_pin_values(offset, pwm_value)
                        log.info(f"[SIM] TARGETS applied: {raw_values}")

                else:
                    log.info(f"[SIM] Received command: {command_string}")
                return

            if not self.serial or not self.serial.is_open:
                raise ConnectionError("ERROR: Serial port not open!")

            self.serial.write((command_string + "\n").encode("utf-8"))
            log.info(f"Command sent to board: {command_string}...")

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
            return max(0, min(MAX_MODULATION_VALUE, int(value)))
        
        return 1 if int(value) else 0

    def set_pin_voltage(self, i: int, value: int):

        if i < 1 or i > 6:
            raise ValueError("ERROR: index must be 1-6!")
        
        if i <= 5:
            value = max(0, min(MAX_MODULATION_VALUE, int(value)))
        else:
            value = 1 if int(value) else 0

        self.send_command(f"PIN {int(i)} {int(value)}")

    def set_pin_voltages(self, voltages):

        if voltages is None:
            raise ValueError("ERROR: Missing voltage targets!")

        try:
            values = [float(v) for v in voltages]
        except Exception as fault:
            raise ValueError(f"ERROR: Invalid voltage target - {fault}!") from fault

        if len(values) < CONTROL_PIN_COUNT:
            raise ValueError(f"ERROR: I need {CONTROL_PIN_COUNT} voltages!")

        trimmed = [self.clamp_voltage(v) for v in values[:CONTROL_PIN_COUNT]]
        payload = "TARGETS " + " ".join(f"{v:.6f}" for v in trimmed)
        self.send_command(payload)

    def set_pwm(self, channel: int, duty: int):

        if channel < 1 or channel > 5:
            raise ValueError("ERROR: PWM channel must be 1-5!")
        
        self.set_pin_voltage(channel, duty)

    # REMOVE ONCE RIG IS FINALISED
    def set_switch(self, on: bool):

        self.set_pin_voltage(6, 1 if on else 0)

    def set_switch_timing_us(self, timing_us: float):
        try:
            value = float(timing_us)
        except Exception:
            return

        board_switch_timing = max(1.0, min(20.0, value))
        self.switch_timing_us = board_switch_timing

        if self.offline:
            try:
                log.info(f"[SIM] Switch timing set to {board_switch_timing:.1f} us")
            except Exception:
                pass
            return

        try:
            self.send_command(f"SWITCH_PERIOD_US {int(board_switch_timing)}")
        except Exception as fault:
            log.error(f"ERROR: Failed to send switch timing '{board_switch_timing}': {fault}")

    def set_pin(self, i: int, value: int):
        self.set_pin_voltage(i, value)

    def set_pin_by_name(self, name: str, value: int):
        i = self._name_to_index(name)
        if not (1 <= i <= 6):
            raise ValueError("ERROR: Unknown pin name!")
        if i <= 5:
            value = max(0, min(MAX_MODULATION_VALUE, int(value)))
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

    def _RNN_training_sweeps(self, minimum_voltage: float, maximum_voltage: float, step: float, hold_time: float, epochs: int, baseline_levels: int | None = None, factorial_levels: int | None = None, random_samples: int | None = None):
        try:
            self.sweep_status = {
                "state": "running",
                "progress": 0.0,
                "message": "",
            }

            try:
                minimum_voltage = float(minimum_voltage)
                maximum_voltage = float(maximum_voltage)
            except Exception:
                minimum_voltage, maximum_voltage = 0.0, 3.3
            if minimum_voltage > maximum_voltage:
                minimum_voltage, maximum_voltage = maximum_voltage, minimum_voltage

            span = max(0.0, maximum_voltage - minimum_voltage)

            try:
                step = float(step)
            except Exception:
                step = 0.05
            if step <= 0:
                step = max(0.01, span / 25.0) if span > 0 else 0.05

            try:
                hold_time = max(0.01, float(hold_time))
            except Exception:
                hold_time = 0.05

            grid = np.arange(minimum_voltage, maximum_voltage + 1e-9, step, dtype=float)

            if grid.size == 0:
                grid = np.array([minimum_voltage], dtype=float)
            grid = np.clip(grid, minimum_voltage, maximum_voltage)

            try:
                baseline_count = int(baseline_levels) if baseline_levels is not None else (3 if span > 0 else 1)
            except Exception:
                baseline_count = 3 if span > 0 else 1
            baseline_count = max(1, baseline_count)
            baseline_levels = np.linspace(minimum_voltage, maximum_voltage, num=baseline_count, dtype=float)
            baseline_levels = np.unique(np.round(baseline_levels, 6))

            if baseline_levels.size == 0:
                baseline_levels = np.array([minimum_voltage], dtype=float)

            try:
                factorial_level_count = int(factorial_levels) if factorial_levels is not None else (3 if span > 0 else 1)
            except Exception:
                factorial_level_count = 3 if span > 0 else 1
            factorial_level_count = max(1, factorial_level_count)
            factorial_levels = np.linspace(minimum_voltage, maximum_voltage, num=factorial_level_count, dtype=float)
            factorial_levels = np.clip(factorial_levels, minimum_voltage, maximum_voltage)

            if random_samples is None:
                random_sample_count = max(int(len(grid) * CONTROL_PIN_COUNT), 20)
            else:
                try:
                    random_sample_count = int(random_samples)
                except Exception:
                    random_sample_count = 0
                if random_sample_count <= 0:
                    random_sample_count = max(int(len(grid) * CONTROL_PIN_COUNT), 20)

            stage1_steps = int(len(baseline_levels) * CONTROL_PIN_COUNT * len(grid))
            stage2a_steps = int(len(factorial_levels) ** CONTROL_PIN_COUNT)
            total_steps = stage1_steps + stage2a_steps + random_sample_count

            if total_steps <= 0:
                total_steps = 1

            rng = np.random.default_rng()
            completed = 0

            class SweepAbort(Exception):
                pass

            def run_step(target_vector, stage_label):
                nonlocal completed

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
                            run_step(
                                targets,
                                f"Sweep 1: pin {pin_index + 1} sweep @ baseline {baseline:.2f} V",
                            )

                combo_total = max(1, stage2a_steps)
                combo_index = 0

                for combo in itertools.product(factorial_levels, repeat=CONTROL_PIN_COUNT):
                    combo_index += 1
                    run_step(combo, f"Sweep 2: factorial sweep for the RNN ({combo_index}/{combo_total})")

                for sample_index in range(1, random_sample_count + 1):
                    random_targets = rng.uniform(minimum_voltage, maximum_voltage, CONTROL_PIN_COUNT)
                    run_step(
                        random_targets.tolist(),
                        f"Stage 3: random sampling for the RNN ({sample_index}/{random_sample_count})",
                    )

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
                        available_columns = [c for c in ordered_columns if c in data_frame.columns]
                        dataset = data_frame.loc[:, available_columns].rename(
                            columns={
                                "pin_1": "squeeze_plate",
                                "pin_2": "ion_source",
                                "pin_3": "wein_filter",
                                "pin_4": "cone_1",
                                "pin_5": "cone_2",
                            }
                        )
                        ts = time.strftime("%Y%m%d_%H%M%S")
                        path = f"sweep_dataset_{ts}.csv"
                        dataset.to_csv(path, index=False)
                        log.info(f"Sweep dataset saved to {path}")
                    except Exception as fault:
                        log.error(f"ERROR: Failed to save sweep dataset - {fault}!")

                if train_model is not None:
                    try:
                        train_model(data_frame, number_of_epochs=int(max(1, epochs)))
                        self.last_training_ts = time.time()

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
                        "message": f"Data sweep complete (wihtout training) - {len(data_frame)} samples collected!",
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
        min_v: float = 0.0,
        max_v: float = 3.3,
        step: float = 0.05,
        dwell_s: float = 0.05,
        epochs: int = 10,
        baseline_levels: int | None = None,
        factorial_levels: int | None = None,
        random_samples: int | None = None,
    ):

        if self.sweep_thread and self.sweep_thread.is_alive():
            log.info("Sweep already running; ignoring start request.")
            return False
        
        self.sweep_cancel.clear()
        self.sweep_status = {"state": "queued", "progress": 0.0, "message": ""}
        self.sweep_thread = threading.Thread(
            target=self._RNN_training_sweeps,
            args=(min_v, max_v, step, dwell_s, epochs, baseline_levels, factorial_levels, random_samples),
            daemon=True,
        )
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
set_pin_voltages = Back_End_Controller.set_pin_voltages
set_pwm = Back_End_Controller.set_pwm
set_switch = Back_End_Controller.set_switch
set_switch_timing_us = getattr(Back_End_Controller, "set_switch_timing_us", None)
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
