    # ----------    Backend Communication Layer for ESP32 System   ----------#

    # Handles serial communication between the ESP32 and the Python environment.
    # Provides live data streaming, thread-safe buffering, and command transmission.

    # Designed for concurrent use by:
    # - python_logging_script.py  (data logging)
    # - python_dashboard_script.py (real-time control)

    # -----------------------------------------------------------------------#

import serial
import threading
import queue
import time
import pandas as pd
import logging

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

# -------------------------------------------------------------------------
#                             Backend Class
# -------------------------------------------------------------------------

class SerialBackend:

    def __init__(self, port=SERIAL_PORT, baud=BAUD_RATE):
        self.port = port
        self.baud = baud
        self.serial = None
        self.alive = threading.Event()
        self.lines = queue.Queue(maxsize=MAX_QUEUE_SIZE)
        self.data_lock = threading.Lock()
        self.data_frame = pd.DataFrame(columns=["timestamp", "voltage", "raw_message"])

        self.thread = threading.Thread(target=self._run, daemon=True)
        self.alive.set()
        self.thread.start()

    def _run(self):

        while self.alive.is_set():
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

        while self.alive.is_set():
            try:
                log.info("Connecting to ESP12F...")
                self.serial = serial.Serial(self.port, self.baud, timeout=1)
                log.info(f"Connected to ESP12F on {self.port} at {self.baud} baud...")
                return
            except serial.SerialException as Serial_Exception:
                log.warning(f"Connection failed: {Serial_Exception}. Retrying in {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)

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
            if not self.serial or not self.serial.is_open:
                raise ConnectionError("Serial port not open.")
            self.serial.write((cmd + "\n").encode("utf-8"))
            log.info(f"Sent command: {cmd}")
        except Exception as e:
            log.error(f"Failed to send command '{cmd}': {e}")

    def get_data(self) -> pd.DataFrame:
        
        with self.data_lock:
            return self.data_frame.copy()

    def stop(self):

        log.info("Backend thread stopping...")
        self.alive.clear()
        self._disconnect()
        log.info("Backend thread stopped...")

# -------------------------------------------------------------------------
#                   Global event for board access
# -------------------------------------------------------------------------

reader = SerialBackend()

get_data = reader.get_data
send_command = reader.send_command
lines = reader.lines

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
