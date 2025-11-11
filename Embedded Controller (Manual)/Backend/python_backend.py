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
        self.data_frame = pd.DataFrame(columns=["timestamp", "voltage", "raw_message"])

        self.offline = offline
        self.target_voltage = 1.0  
        self.pins = [0, 0, 0, 0, 0, 0] 
        self.pins_timestamp = 0.0

        self.thread = threading.Thread(target=self._run, daemon=True)
        self.alive.set()
        self.thread.start()

    @staticmethod
    def _name_to_index(token: str) -> int:
        if token is None:
            return 0
        t = str(token).strip().lower()
        mapping = {
            "squeeze_plate": 1,
            "ion_source": 2,
            "wein_filter": 3,
            "cone_1": 4,
            "cone_2": 5,
            "switch_logic": 6,
        }
        if t in mapping:
            return mapping[t]
        try:
            v = int(t)
            return v if 1 <= v <= 6 else 0
        except Exception:
            return 0

    def _run(self):

        while self.alive.is_set():
            if self.offline:
                timestamp = time.time()
                noise = (random.random() - 0.5) * 0.5 
                v = max(0.0, min(3.3, self.target_voltage + noise))
                line = f"MEASURED {v:.3f}"
                if not self.lines.full():
                    self.lines.put((timestamp, line))
                with self.data_lock:
                    self.data_frame.loc[len(self.data_frame)] = {
                        "timestamp": timestamp,
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
                            "voltage": voltage,
                            "raw_message": line,
                        }
                        self.data_frame = self.data_frame.tail(1000).reset_index(drop=True)
                elif line.startswith("PINS"):
                    try:
                        for part in line.split()[1:]:
                            if "=" not in part:
                                continue
                            name, sval = part.split("=", 1)
                            try:
                                val = int(float(sval))
                            except ValueError:
                                continue
                            idx = self._name_to_index(name)
                            self._update_pin_cache(idx, val)
                    except Exception:
                        pass
                elif line.startswith("ACK PIN"):
                    try:
                        tokens = line.split()
                        self._update_pin_cache(int(tokens[2]), int(tokens[3]))
                    except Exception:
                        pass

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
                        self.target_voltage = max(0.0, min(3.3, v))
                        log.info(f"[SIM] Target voltage set to {self.target_voltage:.3f} V")
                    except ValueError:
                        log.warning(f"[SIM] Invalid SET value in command: {cmd}")
                elif len(parts) == 3 and parts[0].upper() == "PIN":
                    try:
                        idx = self._name_to_index(parts[1])
                        val = int(parts[2])
                        if self._update_pin_cache(idx, val):
                            log.info(f"[SIM] PIN {idx} set to {self.pins[idx-1]}")
                        else:
                            log.warning(f"[SIM] PIN index out of range: {idx}")
                    except ValueError:
                        log.warning(f"[SIM] Invalid PIN value in command: {cmd}")
                else:
                    log.info(f"[SIM] Received command: {cmd}")
                return

            if not self.serial or not self.serial.is_open:
                raise ConnectionError("Serial port not open.")
            self.serial.write((cmd + "\n").encode("utf-8"))
            log.info(f"Sent command: {cmd}")
        except Exception as e:
            log.error(f"Failed to send command '{cmd}': {e}")

    def set_pin_voltage(self, index: int, value: int):
        
        if index < 1 or index > 6:
            raise ValueError("ERROR: index must be 1-6!")
        if index <= 5:
            value = max(0, min(1023, int(value)))
        else:
            value = 1 if int(value) else 0
        self.send_command(f"PIN {int(index)} {int(value)}")

    def set_pwm(self, channel: int, duty: int):
        if channel < 1 or channel > 5:
            raise ValueError("ERROR: PWM channel must be 1-5!")
        self.set_pin_voltage(channel, duty)

    def set_switch(self, on: bool):
        self.set_pin_voltage(6, 1 if on else 0)

    def get_pin_value(self, idx: int, value: int):
        if not (1 <= idx <= 6):
            return None
        if idx <= 5:
            return max(0, min(1023, int(value)))
        return 1 if int(value) else 0

    def update_pins(self, idx: int, value: int) -> bool:
        cv = self._coerce_pin_value(idx, value)
        if cv is None:
            return False
        self.pins[idx - 1] = cv
        self.pins_timestamp = time.time()
        return True

    def set_pin(self, index: int, value: int):
        self.set_pin_voltage(index, value)

    def set_pin_by_name(self, name: str, value: int):
        idx = self._name_to_index(name)

        if not (1 <= idx <= 6):
            raise ValueError("ERROR: Unknown pin name!")
        if idx <= 5:
            value = max(0, min(1023, int(value)))
        else:
            value = 1 if int(value) else 0
        self.send_command(f"PIN {str(name)} {int(value)}")

    def set_squeeze_plate(self, duty: int):
        self.set_pin_by_name("squeeze_plate", duty)

    def set_ion_source(self, duty: int):
        self.set_pin_by_name("ion_source", duty)

    def set_wein_filter(self, duty: int):
        self.set_pin_by_name("wein_filter", duty)

    def set_cone_1(self, duty: int):
        self.set_pin_by_name("cone_1", duty)

    def set_cone_2(self, duty: int):
        self.set_pin_by_name("cone_2", duty)

    def set_switch_logic(self, on: bool):
        self.set_pin_by_name("switch_logic", 1 if on else 0)

    def get_data(self) -> pd.DataFrame:
        
        with self.data_lock:
            return self.data_frame.copy()

    def get_status(self) -> str:
        if self.offline:
            return "Simulated"
        if self.serial and getattr(self.serial, "is_open", False):
            return f"Connected ({self.port})"
        return "Connecting..."

    def stop(self):

        log.info("Backend thread stopping...")
        self.alive.clear()
        self._disconnect()
        log.info("Backend thread stopped...")

    # ------------------------------------------------------------------
    #                             Pin status
    # ------------------------------------------------------------------

    def get_pins(self):
        names = [
            "squeeze_plate",
            "ion_source",
            "wein_filter",
            "cone_1",
            "cone_2",
            "switch_logic",
        ]
        return {
            "names": names,
            "values": list(self.pins),
            "timestamp": self.pins_timestamp,
        }

# -------------------------------------------------------------------------
#                   Global event for board access
# -------------------------------------------------------------------------

reader = SerialBackend(offline=OFFLINE)

get_data = reader.get_data
send_command = reader.send_command
set_pin_voltage = reader.set_pin_voltage
set_pwm = reader.set_pwm
set_switch = reader.set_switch
set_pin = reader.set_pin
set_pin_by_name = reader.set_pin_by_name
set_squeeze_plate = reader.set_squeeze_plate
set_ion_source = reader.set_ion_source
set_wein_filter = reader.set_wein_filter
set_cone_1 = reader.set_cone_1
set_cone_2 = reader.set_cone_2
set_switch_logic = reader.set_switch_logic
get_pins = reader.get_pins
lines = reader.lines
get_status = reader.get_status

# -------------------------------------------------------------------------
#                             Manual testing 
# -------------------------------------------------------------------------

if __name__ == "__main__":
    log.info("Starting backend in standalone mode...")
    try:
        while True:
            time.sleep(1)
            df = reader.get_data()
            if not df.empty:
                log.info(f"Latest: {df['voltage'].iloc[-1]:.3f} V ({len(df)} samples)")
    except KeyboardInterrupt:
        reader.stop()
