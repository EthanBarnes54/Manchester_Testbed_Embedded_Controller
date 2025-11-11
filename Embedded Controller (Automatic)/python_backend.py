    # ----------    Backend Communication Layer for ESP32 System   ----------#

    # Handles serial communication between the ESP32 and the Python environment.
    # Provides live data streaming, thread-safe buffering, and command transmission.

    # Designed for use with:
    # - python_logging_script.py  (data logging)
    # - python_dashboard_script.py (real-time control)

    # -----------------------------------------------------------------------#

import serial
import threading
import queue
import time
import pandas as pd
import logging
import os
import random
import numpy as np

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

SERIAL_PORT = "COM4"      # Adjust for OTA

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
        self.current_dac = 1.0

        self.sweep_thread = None
        self.sweep_status = {"state": "idle", "progress": 0.0, "message": ""}
        self.sweep_cancel = threading.Event()

        self.thread = threading.Thread(target=self._run, daemon=True)
        self.alive.set()
        self.thread.start()

    def _run(self):

        while self.alive.is_set():
        
            if self.offline: #Offline mode for testing without hardware
                
                timestamp = time.time()
                noise = (random.random() - 0.5) * 0.5 
                v = max(0.0, min(3.3, self.target_voltage + noise))
                line = f"MEASURED {v:.3f}"
                if not self.lines.full():
                    self.lines.put((timestamp, line))
                with self.data_lock:
                    self.data_frame.loc[len(self.data_frame)] = {
                        "timestamp": timestamp,
                        "dac": float(self.current_dac),
                        "voltage": v,
                        "raw_message": line,
                    }
                    self.data_frame = self.data_frame.tail(1000).reset_index(drop=True)
                time.sleep(0.05)  
                continue

            if self.serial is None or not self.serial.is_open:
                self._connect()

            try: 
                line = self.serial.readline().decode("utf-8", "ignore").strip()
                if not line:
                    continue

                timestamp = time.time()

                if not self.lines.full():
                    self.lines.put((timestamp, line))

                if line.startswith("MEASURED"):
                    try:
                        voltage = float(line.split()[1])
                    except (IndexError, ValueError):
                        voltage = None

                    with self.data_lock:
                        self.data_frame.loc[len(self.data_frame)] = {
                            "timestamp": timestamp,
                            "dac": float(self.current_dac),
                            "voltage": voltage,
                            "raw_message": line,
                        }
                        self.data_frame = self.data_frame.tail(1000).reset_index(drop=True)

            except serial.SerialException as Serial_Exception:

                log.warning(f"Serial exception: {Serial_Exception}. Reconnecting...")
                self._disconnect()
                time.sleep(RETRY_DELAY)

            except Exception as py_Exception:

                log.error(f"Unexpected error: {py_Exception}")
                time.sleep(0.5)

    # ------------------------------------------------------------------
    #                    Connect / Disconnect Switches
    # ------------------------------------------------------------------

    def _connect(self):

        if self.offline:
            
            time.sleep(0.1)
            return

        while self.alive.is_set():
            try:
                log.info("Connecting to ESP12F...")
                self.serial = serial.Serial(self.port, self.baud, timeout=1)
                log.info(f"Connected to ESP12F on {self.port} at {self.baud} baud...")
                return
            except serial.SerialException as Serial_Exception:
                
                log.warning(
                    f"Connection failed: {Serial_Exception}. Falling back to offline..."
                )
                self.offline = True
                return

    def _disconnect(self):

        if self.serial and self.serial.is_open:
            try:
                log.info("Serial closing...")
                self.serial.close()
                log.info("Serial now closed...")
            except Exception:
                pass
        self.serial = None

    # ------------------------------------------------------------------
    #                               Methods
    # ------------------------------------------------------------------

    def send_command(self, cmd: str):

        try:
            if self.offline:
                parts = cmd.strip().split()
                if len(parts) == 2 and parts[0].upper() == "SET":
                    try:
                        v = float(parts[1])
                        v_clamped = max(0.0, min(3.3, v))
                        self.target_voltage = v_clamped
                        self.current_dac = v_clamped
                        log.info(f"[SIM] Target voltage set to {self.target_voltage:.3f} V")
                    except ValueError:
                        log.warning(f"[SIM] Invalid SET value in command: {cmd}")
                else:
                    log.info(f"[SIM] Received command: {cmd}")
                return

            if not self.serial or not self.serial.is_open:
                raise ConnectionError("Serial port not open.")
            self.serial.write((cmd + "\n").encode("utf-8"))
            log.info(f"Sent command: {cmd}")

            parts = cmd.strip().split()
            if len(parts) == 2 and parts[0].upper() == "SET":
                try:
                    v = float(parts[1])
                    self.current_dac = max(0.0, min(3.3, v))
                except ValueError:
                    pass
        except Exception as exception:
            log.error(f"Failed to send command '{cmd}': {exception}")

    def get_data(self) -> pd.DataFrame:
        
        with self.data_lock:
            return self.data_frame.copy()

    def get_status(self) -> str:

        if self.offline:
            return "Simulated"
        if self.serial and getattr(self.serial, "is_open", False):
            return f"Connected via {self.port}..."
        return "Connecting..."

    def stop(self):

        log.info("Backend thread stopping...")
        self.alive.clear()
        self._disconnect()
        log.info("Backend thread stopped...")

    # ------------------------------------------------------------------
    #                         Training sweep control
    # ------------------------------------------------------------------

    def _sweep_worker(self, min_v: float, max_v: float, step: float, dwell_s: float, epochs: int):
        try:
            self.sweep_status = {"state": "running", "progress": 0.0, "message": ""}
            grid = np.arange(min_v, max_v + 1e-9, step, dtype=float)
            total = len(grid)

            for i, v in enumerate(grid):
                if self.sweep_cancel.is_set():
                    self.sweep_status.update({
                        "state": "aborted",
                        "message": "Sweep aborted by user.",
                    })
                    return
                self.send_command(f"SET {v:.3f}")
                time.sleep(dwell_s)
                self.sweep_status["progress"] = (i + 1) / max(total, 1)

            # Snapshot data and train
            if self.sweep_cancel.is_set():
                self.sweep_status.update({
                    "state": "aborted",
                    "message": "Sweep aborted before training.",
                })
                return
            data_frame = self.get_data()
            try:
                import python_RNN_controller as rnnc

                if not data_frame.empty:

                    if "dac" not in data_frame.columns and "voltage" in data_frame.columns:
                        data_frame = data_frame.assign(dac=self.current_dac)
                    rnnc.train_model(data_frame, number_of_epochs=epochs)
                    self.sweep_status.update({
                        "state": "completed",
                        "message": f"Trained on {len(data_frame)} samples.",
                        "progress": 1.0,
                    })

                else:
                    self.sweep_status.update({
                        "state": "failed",
                        "message": "No data collected during sweep.",
                        "progress": 0.0,
                    })

            except Exception as fault:
                self.sweep_status.update({
                    "state": "failed",
                    "message": f"Training error: {fault}",
                })
        except Exception as fault:
            self.sweep_status.update({
                "state": "failed",
                "message": f"Sweep error: {fault}",
            })

    def start_training_sweep(self, min_v: float = 0.0, max_v: float = 3.3, step: float = 0.05, dwell_s: float = 0.05, epochs: int = 10):

        if self.sweep_thread and self.sweep_thread.is_alive():
            log.info("Sweep already running; ignoring start request.")
            return False
        # Reset cancel and status
        self.sweep_cancel.clear()
        self.sweep_status = {"state": "queued", "progress": 0.0, "message": ""}
        self.sweep_thread = threading.Thread(
            target=self._sweep_worker,
            args=(min_v, max_v, step, dwell_s, epochs),
            daemon=True,
        )

        self.sweep_thread.start()
        return True

    def get_sweep_status(self) -> dict:
        return dict(self.sweep_status)

    def stop_training_sweep(self):
        # Cooperative cancel of the sweep worker
        self.sweep_cancel.set()

# -------------------------------------------------------------------------
#                   Global event for board access
# -------------------------------------------------------------------------

reader = SerialBackend(offline=OFFLINE)

get_data = reader.get_data
send_command = reader.send_command
lines = reader.lines
get_status = reader.get_status
start_training_sweep = reader.start_training_sweep
get_sweep_status = reader.get_sweep_status
stop_training_sweep = reader.stop_training_sweep

# -------------------------------------------------------------------------
#                             Manual testing 
# -------------------------------------------------------------------------

if __name__ == "__main__":
    log.info("Starting backend in standalone mode...")
    try:
        while True:
            time.sleep(1)
            data_frame = reader.get_data()
            if not data_frame.empty:
                log.info(f"Latest: {data_frame['voltage'].iloc[-1]:.3f} V ({len(data_frame)} samples)")
    except KeyboardInterrupt:
        reader.stop()
