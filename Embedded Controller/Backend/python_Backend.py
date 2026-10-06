"""
# --------- Backend Communication Layer - ESP32 Cntrol System --------- #

# Handles serial communication between the ESP32 board and the Python environment.
# Provides live data streaming, thread-safe buffering, and command
# transmission. Designed for use with the data loggier, dashboard, signal
# pipeline, and RNN controller modules.

#------------------------------------------------------------------------#
"""
# ---------------------------------------------------------------------- #
#                               Imports                                  #
# ---------------------------------------------------------------------- #

import binascii
import itertools
import logging
from collections import deque
import os
import queue
import random
import threading
import time

import numpy as np
import pandas as pd
import serial

from python_ML_Metrics import MetricCollector
from python_Autonomy_Guard import AutonomyGuard

try:

    from python_RNN_Controller import (
        train_model,
        online_update,
        set_learning_rate as _set_rnn_learning_rate,
        get_learning_rate as _get_rnn_learning_rate,
        set_momentum as _set_rnn_momentum,
        get_momentum as _get_model_momentum,
        set_optimiser_type as _set_rnn_optimiser_type,
        get_optimiser_type as _get_rnn_optimiser_type,
        save_nn_weights,
        model,
        scaler,
        compute_feature_saliencies,
        propose_control_vector,
        get_validation_metrics,
        get_training_feature_stats,
    )

except Exception:
    logging.warning("WARNING: RNN controller module import failed - online learning, controll and training will be unavailable!")

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

    def _set_rnn_optimiser_type(_value):
        return None

    def _get_rnn_optimiser_type():
        return None

    def save_nn_weights(_model, _scaler):
        raise RuntimeError("ERROR: Manual save unavailable! (RNN controller import failed...)")

    model = None
    scaler = None

    def compute_feature_saliencies(_df, max_samples=200):
        raise RuntimeError("ERROR: Feature saliencies unavailable! (RNN controller import failed...)")

    propose_control_vector = None

    def get_validation_metrics():
        return {}

    def get_training_feature_stats():
        return None


# ------------------------------------------------------------------------- # 
#                                 Logging                                   # 
# ------------------------------------------------------------------------- # 

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S",)
log = logging.getLogger("ESP32_Backend")

# -------------------------------------------------------------------------
#                             Configuration
# -------------------------------------------------------------------------

def _configured_serial_port() -> str:
    """Same variable platformio.ini reads for upload_port, so one setting covers flashing
    and running. COM4 is only the fallback for the original lab machine."""

    return os.getenv("TESTBED_SERIAL_PORT", "").strip() or "COM4"


def _env_flag(name: str) -> bool:
    """Reads an on/off environment variable. Case folded so that OFFLINE=FALSE cannot
    quietly turn simulation on."""

    return os.getenv(name, "").strip().lower() not in ("", "0", "false", "no", "off")


SERIAL_PORT = _configured_serial_port()
BAUD_RATE = 115200
RETRY_DELAY = 3.0
MAX_QUEUE_SIZE = 2000

OFFLINE = _env_flag("OFFLINE")

# Alter once board design has been finalised 

MAX_MODULATION_VALUE = 1023
MAX_CONTROL_VOLTAGE = 3.3
CONTROL_PIN_COUNT = 5

# Must mirror SWITCH_PERIOD_MIN_US / SWITCH_PERIOD_MAX_US in the firmware. Both are the
# time between edges, half a cycle, so the 1 us floor is a 500 kHz square wave. It is
# the design floor, not yet scoped on hardware.
SWITCH_PERIOD_MIN_US = 1
SWITCH_PERIOD_MAX_US = 2000000

# Must match PROTOCOL_VERSION in the firmware. The backend will not arm a board whose
# VERSION reply reports a different one.
PROTOCOL_VERSION = 2

# How long a command may go unanswered before it counts as a missed reply.
ACK_TIMEOUT_SEC = 1.0

# What each command is answered with, keyed by its first word. The board answers commands
# in the order it received them, so the oldest outstanding command owns the next line
# that matches it, or the next ERROR.
COMMAND_REPLIES = {
    "PING": ("OK",),
    "VERSION": ("VERSION ",),
    "ARM": ("ACK ARM",),
    "DISARM": ("ACK DISARM",),
    "FAULTS": ("FAULTS ",),
    "CLEAR": ("ACK CLEAR",),
    "SELFTEST": ("SELFTEST ",),
    "HEALTH": ("HEALTH ",),
    "PINS": ("PINS ",),
    "GET": ("PINS ",),
    "TARGETS": ("ACK TARGETS",),
    "PIN": ("ACK PIN",),
    "SWITCH_PERIOD_US": ("ACK SWITCH_PERIOD_US",),
}

# ERROR lines the board sends on its own, which never answer a command.
UNSOLICITED_ERRORS = ("ERROR: ADC conversion timed out", "ERROR: OTA")

# Lines that are part of the protocol, as opposed to free text such as Wi-Fi status. Once
# the board has confirmed the protocol, these are only trusted with a valid CRC.
PROTOCOL_PREFIXES = ("MEASURED", "PINS", "ACK", "ERROR", "OK", "FAULT", "FAILSAFE", "HEALTH", "SELFTEST", "VERSION")


def frame_line(text: str) -> str:
    """Appends the link's CRC, "*XXXX" (CRC-16/CCITT-FALSE), exactly as the firmware does."""

    return f"{text}*{binascii.crc_hqx(text.encode('utf-8'), 0xFFFF):04X}"


def unframe_line(line: str) -> tuple[str, str]:
    """Splits off and checks a line's CRC. Returns (text, "valid" | "invalid" | "absent")."""

    if len(line) >= 5 and line[-5] == "*" and all(digit in "0123456789abcdefABCDEF" for digit in line[-4:]):
        text = line[:-5]
        expected = binascii.crc_hqx(text.encode("utf-8"), 0xFFFF)
        return text, ("valid" if int(line[-4:], 16) == expected else "invalid")

    return line, "absent"


def _command_word(command: str) -> str:
    tokens = command.strip().split()
    return tokens[0].upper() if tokens else ""


# Keepalive cadence, kept well inside the firmware's COMMAND_TIMEOUT_MS so that a
# quiet but healthy link is never mistaken for a dead host.
KEEPALIVE_INTERVAL_SEC = 2.0

ONLINE_UPDATE_INTERVAL_SEC = 5.0

# Auto control runs on its own backend thread rather than in a dashboard callback, so
# it keeps its own cadence, carries on with no browser open, and has a single writer
# however many tabs are watching.
AUTO_CONTROL_MIN_PERIOD_MS = 100
AUTO_CONTROL_MAX_PERIOD_MS = 60000
AUTO_CONTROL_DEFAULT_PERIOD_MS = 500
AUTO_CONTROL_DEFAULT_CHANGE_PENALTY = 0.1

# Proposals are only made from fresh hardware readings. If the board goes quiet the
# controller stops steering rather than acting on a picture of the beam that is stale.
AUTO_CONTROL_MAX_SAMPLE_AGE_SEC = 2.0
DEFAULT_UPDATE_WINDOW = 30.0
UPDATE_WINDOW_TIME = 5.0
ONLINE_WINDOW_MAX_SEC = 600.0

DEFAULT_DATA_BUFFER_SAMPLES = 1000
DATA_BUFFER_MIN_SAMPLES = 100
DATA_BUFFER_MAX_SAMPLES = 100000

# A sweep records into its own capture rather than the rolling buffer, which is far
# smaller than a sweep. This only bounds memory if a sweep is configured absurdly large.
SWEEP_CAPTURE_MAX_SAMPLES = DATA_BUFFER_MAX_SAMPLES

# Provenance tags - every buffered sample is stamped with where it came from, so a
# simulated trace can never be mistaken for a diode reading after the fact.
HARDWARE_SOURCE = "hardware"
SIMULATED_SOURCE = "simulated"


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

        """Creates the backend. Nothing touches the port or starts a thread until start().

        Arguments:
            port: Serial port name (for example, "COM4").
            baud: Serial baud rate.
            status: When True, starts in offline/simulated mode.
        """

        self.port = port
        self.baud = baud
        self.serial = None

        self.alive = threading.Event()
        self._stop_event = threading.Event()
        self.lines = queue.Queue(maxsize=MAX_QUEUE_SIZE)
        self.data_lock = threading.Lock()
        self.command_lock = threading.Lock()

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
                "source",
            ]
        )

        self.max_buffer_samples = DEFAULT_DATA_BUFFER_SAMPLES

        self.force_offline = bool(status)
        self.offline = False
        self.last_connection_attempt = 0.0
        self._set_offline_state(status)

        self.pins = [0, 0, 0, 0, 0, 0]
        self.pins_timestamp = 0.0
        self.switch_timing = None

        # The board's safety state as it last reported it. UNKNOWN until it has said, and
        # again whenever the link drops. A simulated board boots SAFE like a real one.
        self.board_mode = "SAFE" if self.force_offline else "UNKNOWN"
        self.board_faults = {}
        self.last_board_fault = None
        self.last_arm_refusal = ""
        self.board_health = {}
        self.last_selftest = None

        # The link: what the board says it is, whether it speaks this protocol (None until it
        # has answered VERSION), what is waiting for a reply, and running counts.
        self.board_version = {}
        self.protocol_ok = True if self.force_offline else None
        self._pending_replies = deque()
        self._pending_lock = threading.Lock()
        self._last_measured_seq = None
        self.link_stats = dict.fromkeys(
            ("lines", "bad_checksums", "unverified", "measured", "measured_missing", "board_restarts",
             "replies", "rejections", "reply_timeouts", "queue_drops"), 0)

        self.sweep_thread = None
        self.sweep_status = {"state": "idle", "progress": 0.0, "message": ""}
        self.sweep_cancel = threading.Event()
        self._sweep_capture = None

        self.save_dataset_enabled = False
        self.last_training_time_stamp = None
        self.last_update_time_stamp = None
        self.online_window_seconds = DEFAULT_UPDATE_WINDOW
        self.online_update_period = ONLINE_UPDATE_INTERVAL_SEC
        self.online_update_enabled = True

        self.auto_control_enabled = False
        self.auto_control_period_ms = AUTO_CONTROL_DEFAULT_PERIOD_MS
        self.auto_control_change_penalty = AUTO_CONTROL_DEFAULT_CHANGE_PENALTY
        self.auto_control_status = {"state": "off", "message": "", "last_update": None}

        # Runtime assurance between the model and the board (python_Autonomy_Guard).
        self.autonomy_guard = AutonomyGuard()

        self.thread = None
        self.online_update_thread = None
        self.keepalive_thread = None
        self.auto_control_thread = None

    def start(self):
        """Opens the link and starts the reader, online update and keepalive threads.

        Kept out of the constructor so that importing this module, or building a
        backend in a test, never opens a serial port. Calling it again is a no-op.
        """

        if self.alive.is_set():
            return self

        self._stop_event.clear()
        self.alive.set()

        self.thread = threading.Thread(target=self.operating_system, name="backend-serial", daemon=True)
        self.online_update_thread = threading.Thread(target=self._update_manager, name="backend-online-update", daemon=True)
        self.keepalive_thread = threading.Thread(target=self._keepalive_manager, name="backend-keepalive", daemon=True)
        self.auto_control_thread = threading.Thread(target=self._auto_control_manager, name="backend-auto-control", daemon=True)

        for worker in self._workers():
            worker.start()

        return self

    def _workers(self):
        """Returns the background threads that start() owns."""

        workers = (self.thread, self.online_update_thread, self.keepalive_thread, self.auto_control_thread)
        return [worker for worker in workers if worker is not None]

    # ------------------------------------------------------------------ # 
    #                     Connect / Disconnect Functions                 # 
    # ------------------------------------------------------------------ # 

    def _set_offline_state(self, value: bool):
        """Keep instance and module offline flags in sync."""
        global OFFLINE
        self.offline = bool(value)
        OFFLINE = self.offline

    def connect(self):
        """Attempts one serial connection and updates live connectivity state."""

        if self.force_offline:
            self._set_offline_state(True)
            return False

        if self.serial and self.serial.is_open:
            self._set_offline_state(False)
            return True

        try:
            log.info("Connecting to ESP32...")
            self.serial = serial.Serial(self.port, self.baud, timeout=1)
            self._set_offline_state(False)
            log.info(f"Connected to ESP32 on {self.port} at {self.baud} baud...")

            # Nothing is armed until the board has said which protocol it speaks.
            self.send_command("VERSION")
            return True

        except serial.SerialException as fault:
            self.serial = None
            self._set_offline_state(True)
            log.warning(f"ERROR: Connection to board failed - {fault}! Operating in offline mode...")
            return False

        except Exception as fault:
            self.serial = None
            self._set_offline_state(True)
            log.warning(f"ERROR: Unexpected connection fault - {fault}! Operating in offline mode...")
            return False

    def disconnect(self):
        """Closes the serial connection to the control board."""

        if self.serial and self.serial.is_open:
            try:
                log.info("Serial closing...")
                self.serial.close()
                log.info("Serial closed!")

            except Exception:
                pass

        self.serial = None
        self._set_offline_state(True)

        if not self.force_offline:
            self.board_mode = "UNKNOWN"
            self.protocol_ok = None
            self.board_version = {}
            self._last_measured_seq = None

            with self._pending_lock:
                self._pending_replies.clear()

    def set_save_dataset_enabled(self, enabled: bool):
        """Toggles dataset saving."""

        try:
            self.save_dataset_enabled = bool(enabled)

        except Exception:
            self.save_dataset_enabled = False

    def set_window_update_time(self, window_seconds: float):
        """Sets the ML learning update window length.

        Arguments:
            window_seconds: Desired window length in seconds.

        Returns:
            The clamped window length in seconds.
        """
        try:
            window_time = float(window_seconds)

        except Exception as fault:
            raise ValueError(f"ERROR: Invalid time window '{window_seconds}' : {fault}!")
        
        window_bounds = max(UPDATE_WINDOW_TIME, min(ONLINE_WINDOW_MAX_SEC, window_time))
        self.online_window_seconds = window_bounds

        log.info(f"Update window duration set to {window_bounds:.1f} s...")

        return window_bounds
    
    def get_window_update_time(self) -> float:
        """Retrieves the current online-learning update window length."""

        return float(self.online_window_seconds)

    def set_online_learning_rate(self, learning_rate: float):
        """Sets the RNN online-learning rate."""

        if _set_rnn_learning_rate is None:
            raise RuntimeError("ERROR: Learning rate unavailable for upload! (RNN controller import failed...)")
        return _set_rnn_learning_rate(learning_rate)

    def get_learning_rate(self):
        """Returns the current RNN learning rate."""

        if _get_rnn_learning_rate is None:
            log.warning("WARNING: Cannot retrieve learning rate, continuing without momentum...")
            return None
        
        return _get_rnn_learning_rate()

    def set_model_momentum(self, momentum: float):
        """Sets the RNN learning momentum."""

        if _set_rnn_momentum is None:
            raise RuntimeError("ERROR: Momentum unavailable for upload! (RNN controller import failed...)")
        
        return _set_rnn_momentum(momentum)

    def get_model_momentum(self):
        """Returns the current learning momentum."""

        if _get_model_momentum is None:
            log.warning("WARNING: Cannot retrieve momentum, continuing without momentum...")

            return None
        
        return _get_model_momentum()

    def set_optimiser_type(self, optimiser_type: str):
        """Sets the optimiser type for the RNN controller."""

        if _set_rnn_optimiser_type is None:
            raise RuntimeError("ERROR: Optimiser selection unavailable! Please try again...")
        
        return _set_rnn_optimiser_type(optimiser_type)

    def get_optimiser_type(self):
        """Returns the current optimiser type."""

        if _get_rnn_optimiser_type is None:
            log.warning("WARNING: Cannot retrieve optimiser type, continuing without explicit knowledge and trusting the RNN...")
            return None
        
        return _get_rnn_optimiser_type()

    def save_model_parameters(self):
        """Saves RNN model parameters to local machine in working folder."""

        if save_nn_weights is None:
            raise RuntimeError("ERROR: Manual save unavailable! (RNN controller import failed...)")
        
        return save_nn_weights(model, scaler)

    def compute_feature_importance(self, max_samples: int = 200, num_permutations: int = 20):
        """Computes feature importances/saliencies from recent backend data, using shapley permutations."""

        data_frame = self.get_training_data()
        return compute_feature_saliencies(data_frame, max_samples=max_samples, num_permutations=num_permutations)

    def get_online_update_config(self) -> dict:
        """Returns a snapshot of the current, online update settings."""

        return {
            "window_seconds": float(self.online_window_seconds),
            "learning_rate": self.get_learning_rate(),
            "enabled": bool(self.online_update_enabled),
            "momentum": self.get_model_momentum(),
        }

    def get_latest_point(self):
        """Returns the most recent (timestamp, voltage) pair from the datastream/pipeline ."""

        with self.data_lock:

            if self.data_frame.empty:
                log.warning("WARNING: No data available in data frame!")
                return None
            
            row = self.data_frame.iloc[-1]
            return float(row["timestamp"]), float(row["voltage"])


    # ------------------------------------------------------------------ #
    #                         Object Methods                             #
    # ------------------------------------------------------------------ #

    
    @staticmethod
    def _name_to_pin_index(pin_name: str) -> int:
        """Maps a pin label (or numeric string) to a 1-based pin index (1-6)."""

        if pin_name is None:
            log.error("ERROR: Pin name is invalid or empty! Please enter a valid pin name/index...")
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
        """Clamps a PWM pulse command to the acceptible range."""

        try:
            signal_value = int(round(float(value)))

        except Exception as fault:
            return log.error(f"ERROR: Pulse signal out of range '{value}' : {fault}!")
        
        return max(0, min(MAX_MODULATION_VALUE, signal_value))

    @staticmethod
    def clamp_voltage(voltage: float) -> float:
        """Clamps a voltage command to the acceptable range."""

        try:
            voltage_value = float(voltage)

        except Exception as fault:
            return log.error(f"ERROR: Voltage out of range '{voltage}' : {fault}!")
        
        return max(0.0, min(MAX_CONTROL_VOLTAGE, voltage_value))

    @classmethod
    def pulse_to_voltage(cls, value: float) -> float:
        """Converts a PWM pulse to volts."""

        pulse_signal = cls._clamp_signal_pulse(value)

        return (pulse_signal / MAX_MODULATION_VALUE) * MAX_CONTROL_VOLTAGE

    @classmethod
    def voltage_to_pulse(cls, voltage: float) -> int:
        """Converts voltage to a PWM pulse."""

        voltage_output = cls.clamp_voltage(voltage)
        scaled_voltage_output = (voltage_output / MAX_CONTROL_VOLTAGE) * MAX_MODULATION_VALUE

        return cls._clamp_signal_pulse(scaled_voltage_output)

    def update_pin_values(self, pin_index: int, pin_value: int) -> bool:
        """Updates cached pin values and timestamp."""

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

    def set_buffer_samples(self, sample_count: int) -> int:
        """Sets the maximum number of samples stored in the rolling buffer."""

        try:
            requested_float = float(sample_count)
            if not requested_float.is_integer():
                raise ValueError("Buffer size must be an integer value")
            requested = int(requested_float)
        except Exception as fault:
            raise ValueError(f"ERROR: Invalid buffer sample count - {fault}!")

        if requested < DATA_BUFFER_MIN_SAMPLES or requested > DATA_BUFFER_MAX_SAMPLES:
            raise ValueError(
                f"ERROR: Buffer sample count out of range ({DATA_BUFFER_MIN_SAMPLES}-{DATA_BUFFER_MAX_SAMPLES})!"
            )

        self.max_buffer_samples = requested

        with self.data_lock:
            if len(self.data_frame) > requested:
                self.data_frame = self.data_frame.tail(requested).reset_index(drop=True)

        return self.max_buffer_samples

    def get_buffer_samples(self) -> int:
        """Returns the current rolling buffer size."""
        return int(self.max_buffer_samples)

    def _append_measurement(self, timestamp: float, voltage: float, message: str, source: str = HARDWARE_SOURCE):
        """Append a measurement row to the internal DataFrame, stamped with its provenance."""

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
            "source": source,
        }

        with self.data_lock:
            self.data_frame.loc[len(self.data_frame)] = row
            self.data_frame = self.data_frame.tail(self.max_buffer_samples).reset_index(drop=True)

            if self._sweep_capture is not None:
                self._sweep_capture.append(row)

    def _begin_sweep_capture(self):
        """Starts recording every sample into a sweep capture that the rolling buffer cannot trim."""

        with self.data_lock:
            self._sweep_capture = deque(maxlen=SWEEP_CAPTURE_MAX_SAMPLES)

    def _end_sweep_capture(self) -> pd.DataFrame:
        """Stops the sweep capture and returns everything it recorded."""

        with self.data_lock:
            captured = list(self._sweep_capture or [])
            self._sweep_capture = None

        if len(captured) >= SWEEP_CAPTURE_MAX_SAMPLES:
            log.warning(f"WARNING: Sweep exceeded {SWEEP_CAPTURE_MAX_SAMPLES} samples, only the most recent were kept!")

        return pd.DataFrame(captured, columns=self.data_frame.columns)

    def operating_system(self):
        """ Reads serial messages, parses measurement / pin snapshots, and updates
        the internal buffers. When offline, generates simulated readings.
        """

        while self.alive.is_set():

            if self.serial is None or not self.serial.is_open:
                now = time.time()

                if now - self.last_connection_attempt >= RETRY_DELAY:
                    self.last_connection_attempt = now
                    self.connect()

            # Purely offline mode - simulate data to test functionality and connectivity.
            if self.offline or self.serial is None or not self.serial.is_open:

                timestamp = time.time()

                analog_snapshot = [self.pulse_to_voltage(p) for p in self.pins[:CONTROL_PIN_COUNT]]
                avg_voltage = float(np.mean(analog_snapshot)) if analog_snapshot else 0.0
                noise = (random.random() - 0.5) * 0.5
                v = max(0.0, min(MAX_CONTROL_VOLTAGE, avg_voltage + noise))

                # Trailing tag keeps the frame parseable while marking it as not a real reading.
                message = f"MEASURED {v:.3f} SIMULATED"

                if not self.lines.full():
                    self.lines.put((timestamp, message))

                self._append_measurement(timestamp, v, message, source=SIMULATED_SOURCE)

                time.sleep(0.05)
                continue

            try:
                message = self.serial.readline().decode("utf-8", "ignore").strip()

                if not message:
                    continue

                timestamp = time.time()
                message = self._accept_line(message)

                if message is None:
                    continue

                if not self.lines.full():
                    self.lines.put((timestamp, message))
                else:
                    self.link_stats["queue_drops"] += 1

                self._handle_board_line(timestamp, message)
                self._match_reply(message)

            except serial.SerialException as serial_fault:
                log.warning(f"ERROR: Serial exception - {serial_fault}! Reconnecting...")
                self.disconnect()
                self._stop_event.wait(RETRY_DELAY)

            except Exception as fault:
                log.error(f"ERROR: Unexpected fault - {fault}! Reconnecting...")
                self._stop_event.wait(0.5)

    def _accept_line(self, raw: str) -> str | None:
        """Checks a received line's CRC. Returns its text, or None if it must be dropped:
        a CRC that does not match, or a protocol line without one once the board has
        confirmed it speaks this protocol. Free text (Wi-Fi status, debug logs) needs none."""

        self.link_stats["lines"] += 1
        text, crc = unframe_line(raw)

        if crc == "invalid":
            self.link_stats["bad_checksums"] += 1
            log.warning(f"WARNING: Dropped a corrupted line from the board: {raw!r}")
            return None

        if crc == "absent" and self.protocol_ok and text.startswith(PROTOCOL_PREFIXES):
            self.link_stats["unverified"] += 1
            log.warning(f"WARNING: Dropped a protocol line with no CRC: {raw!r}")
            return None

        return text

    def _match_reply(self, message: str):
        """Settles the oldest outstanding command if this line answers it."""

        with self._pending_lock:
            if not self._pending_replies:
                return

            _, _, prefixes = self._pending_replies[0]

            if message.startswith(prefixes):
                self._pending_replies.popleft()
                self.link_stats["replies"] += 1

            elif message.startswith("ERROR") and not message.startswith(UNSOLICITED_ERRORS):
                self._pending_replies.popleft()
                self.link_stats["rejections"] += 1

    def _expire_pending_replies(self, now: float | None = None):
        """Counts and drops commands that have waited longer than ACK_TIMEOUT_SEC for a reply."""

        now = time.time() if now is None else now

        with self._pending_lock:
            while self._pending_replies and now - self._pending_replies[0][0] > ACK_TIMEOUT_SEC:
                _, word, _ = self._pending_replies.popleft()
                self.link_stats["reply_timeouts"] += 1
                log.warning(f"WARNING: The board did not answer {word} within {ACK_TIMEOUT_SEC:.1f} s!")

    def _note_measurement_sequence(self, message: str):
        """Counts readings that went missing between two MEASURED lines, and board restarts."""

        self.link_stats["measured"] += 1

        try:
            sequence = int(self._key_values(message)["seq"])
        except (KeyError, ValueError):
            return

        previous = self._last_measured_seq
        self._last_measured_seq = sequence

        if previous is None:
            return

        if sequence > previous + 1:
            self.link_stats["measured_missing"] += sequence - previous - 1

        elif sequence <= previous:
            self.link_stats["board_restarts"] += 1
            log.warning(f"WARNING: MEASURED sequence went from {previous} back to {sequence}, the board has restarted!")

    def _handle_board_line(self, timestamp: float, message: str):
        """Acts on one line from the board: measurements, pin reports and safety state."""

        if message.startswith("MEASURED"):
            self._note_measurement_sequence(message)

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

        else:
            self._handle_safety_line(timestamp, message)

    def _handle_safety_line(self, timestamp: float, message: str):
        """Tracks the board's mode and faults from ACK ARM/DISARM, FAULT(S) and FAILSAFE lines."""

        if message == "ACK ARM":
            self.board_mode = "ARMED"
            self.last_arm_refusal = ""
            log.info("Board ARMED...")

        elif message == "ACK DISARM" or message.startswith("FAILSAFE outputs zeroed"):
            self.board_mode = "SAFE" if self.board_mode != "FAULT" else "FAULT"

        elif message.startswith("ACK CLEAR FAULTS") or message.startswith("FAULTS "):
            fields = self._key_values(message)

            if "mode" in fields:
                self.board_mode = fields["mode"]

            if message.startswith("FAULTS "):
                self.board_faults = fields

        elif message.startswith("FAULT "):
            tokens = message.split()
            fields = self._key_values(message)
            self.last_board_fault = {"time": timestamp, "fault": tokens[1] if len(tokens) > 1 else "",
                                     "severity": tokens[2] if len(tokens) > 2 else ""}

            if "mode" in fields:
                self.board_mode = fields["mode"]

            log.warning(f"WARNING: Board reported {message}")

        elif message.startswith("HEALTH "):
            self.board_health = {"time": timestamp, **self._key_values(message)}
            self.board_mode = self.board_health.get("mode", self.board_mode)

        elif message.startswith("VERSION "):
            self.board_version = self._key_values(message)

            try:
                self.protocol_ok = int(self.board_version.get("protocol", -1)) == PROTOCOL_VERSION
            except ValueError:
                self.protocol_ok = False

            if self.protocol_ok:
                log.info(f"Board firmware {self.board_version.get('firmware')} speaks protocol {PROTOCOL_VERSION}...")
            else:
                log.error(f"ERROR: Board speaks protocol {self.board_version.get('protocol')}, this backend speaks "
                          f"{PROTOCOL_VERSION}! Refusing to arm it.")

        elif message.startswith("SELFTEST "):
            tokens = message.split()
            self.last_selftest = {"time": timestamp, "result": tokens[1] if len(tokens) > 1 else "",
                                  "checks": self._key_values(message)}

            if self.last_selftest["result"] != "PASS":
                log.warning(f"WARNING: Board self-test failed - {message}")

        elif message.startswith("ERROR: Cannot arm"):
            self.last_arm_refusal = message
            log.warning(f"WARNING: {message}")

        elif message.startswith("ERROR: Not armed"):
            log.warning("WARNING: The board refused an output command because it is not armed!")

    @staticmethod
    def _key_values(message: str) -> dict:
        """The key=value fields of a board line."""

        fields = {}

        for token in message.split():
            if "=" in token:
                key, value = token.split("=", 1)
                fields[key] = value

        return fields

    # ------------------------------------------------------------------ #
    #                               Safety                               #
    # ------------------------------------------------------------------ #

    def is_armed(self) -> bool:
        return self.board_mode == "ARMED"

    def arm(self):
        """Asks the board to arm. It refuses while a critical fault is latched, and this
        backend refuses until the board has confirmed it speaks the same protocol."""

        if not self.force_offline and self.protocol_ok is not True:
            reason = ("the board has not answered VERSION yet" if self.protocol_ok is None
                      else f"the board speaks protocol {self.board_version.get('protocol')}, not {PROTOCOL_VERSION}")
            self.last_arm_refusal = f"Backend refused to arm: {reason}"
            log.error(f"ERROR: {self.last_arm_refusal}!")
            return

        self.send_command("ARM")

    def disarm(self):
        """Zeroes every output and closes the switch gate. Always accepted."""

        self.send_command("DISARM")

    def clear_faults(self):
        """Clears latched faults whose condition has gone, and asks for the fault report."""

        self.send_command("CLEAR FAULTS")
        self.send_command("FAULTS")

    def request_faults(self):
        self.send_command("FAULTS")

    def run_selftest(self):
        """Asks the board to rerun its power-on self-test and report the result."""

        self.send_command("SELFTEST")

    def get_link_health(self) -> dict:
        """The link's state: the board's VERSION reply, whether its protocol matches, and counts."""

        with self._pending_lock:
            outstanding = len(self._pending_replies)

        return {"version": dict(self.board_version), "protocol_ok": self.protocol_ok,
                "outstanding_replies": outstanding, **self.link_stats}

    def get_board_health(self) -> dict:
        """The board's last HEALTH report and self-test result."""

        return {"health": dict(self.board_health), "selftest": dict(self.last_selftest) if self.last_selftest else None}

    def get_board_safety(self) -> dict:
        """The board's mode and fault report as last received."""

        return {
            "mode": self.board_mode,
            "faults": dict(self.board_faults),
            "last_fault": self.last_board_fault,
            "last_arm_refusal": self.last_arm_refusal,
        }

    def _keepalive_manager(self):
        """Pings the board on a fixed cadence so its output failsafe stays satisfied.

        The firmware drops its outputs when the host goes quiet. Passive monitoring
        sends nothing on its own, so without this a perfectly healthy but idle
        session would trip the failsafe.
        """

        while self.alive.is_set():

            if self._stop_event.wait(KEEPALIVE_INTERVAL_SEC):
                break

            if self.offline or self.serial is None or not self.serial.is_open:
                continue

            try:
                self._expire_pending_replies()

                if self.protocol_ok is None:
                    self.send_command("VERSION")

                self.send_command("PING")

                # Also keeps the board's mode, faults and health current for the
                # dashboard and for the auto control and sweep checks.
                self.send_command("FAULTS")
                self.send_command("HEALTH")

            except Exception as fault:
                log.warning(f"WARNING: Keepalive ping failed - {fault}!")

    # ------------------------------------------------------------------ #
    #                           Auto control                             #
    # ------------------------------------------------------------------ #

    def set_auto_control(self, enabled: bool | None = None, period_ms: float | None = None, change_penalty: float | None = None) -> dict:
        """Changes any of the auto control settings and returns the resulting configuration."""

        if period_ms is not None:
            try:
                period_value = float(period_ms)

            except Exception as fault:
                raise ValueError(f"ERROR: Invalid auto control period '{period_ms}' - {fault}!")

            self.auto_control_period_ms = max(AUTO_CONTROL_MIN_PERIOD_MS, min(AUTO_CONTROL_MAX_PERIOD_MS, period_value))

        if change_penalty is not None:
            try:
                penalty_value = float(change_penalty)

            except Exception as fault:
                raise ValueError(f"ERROR: Invalid change penalty '{change_penalty}' - {fault}!")

            self.auto_control_change_penalty = max(0.0, penalty_value)

        if enabled and not self.auto_control_enabled:
            admissible, reason = self.autonomy_guard.model_admissible(get_validation_metrics())

            if not admissible:
                self._set_auto_control_status("refused", f"Auto control refused: {reason}")
                log.warning(f"WARNING: Auto control refused - {reason}!")
                return self.get_auto_control()

        if enabled is not None and bool(enabled) != self.auto_control_enabled:
            self.auto_control_enabled = bool(enabled)
            self._set_auto_control_status("starting" if self.auto_control_enabled else "off")
            log.info(f"Auto control {'enabled' if self.auto_control_enabled else 'disabled'}...")

        return self.get_auto_control()

    def get_auto_control(self) -> dict:
        """Returns the auto control settings alongside what the controller is currently doing."""

        counts = self.autonomy_guard.counts

        return {
            "enabled": bool(self.auto_control_enabled),
            "period_ms": float(self.auto_control_period_ms),
            "change_penalty": float(self.auto_control_change_penalty),
            **dict(self.auto_control_status),
            "guard": {"accepted": counts.accepted, "limited": counts.limited, "rejected": counts.rejected,
                      "held_for_drift": counts.held_for_drift},
        }

    def _set_auto_control_status(self, state: str, message: str = ""):
        """Records what the controller is doing, logging only on a change so a 10 Hz loop cannot flood the log."""

        previous = self.auto_control_status

        if message and (previous.get("state") != state or previous.get("message") != message):
            log.info(f"Auto control {state}: {message}")

        self.auto_control_status = {"state": state, "message": message, "last_update": previous.get("last_update")}

    def _auto_control_step(self) -> bool:
        """Makes one control decision. Returns True when new targets were sent to the board."""

        if not self.auto_control_enabled:
            self._set_auto_control_status("off")
            return False

        if propose_control_vector is None:
            self._set_auto_control_status("unavailable", "RNN controller import failed")
            return False

        if str(self.sweep_status.get("state", "")).lower() == "running":
            self._set_auto_control_status("paused", "Training sweep in progress")
            return False

        if not self.is_armed():
            self._set_auto_control_status("waiting", f"Board is not armed ({self.board_mode})")
            return False

        data_frame = self.get_training_data()

        if data_frame.empty:
            self._set_auto_control_status("waiting", "No readings from the board yet")
            return False

        sample_age = time.time() - float(data_frame["timestamp"].iloc[-1])

        if sample_age > AUTO_CONTROL_MAX_SAMPLE_AGE_SEC:
            self._set_auto_control_status("waiting", f"Newest reading is {sample_age:.1f} s old")
            return False

        # Outside the data the model was trained on, its proposals mean nothing: hold.
        drifted, reason = self.autonomy_guard.drift(data_frame, get_training_feature_stats())

        if drifted:
            self._set_auto_control_status("holding", f"Input drift - {reason}")
            return False

        try:
            # TODO: Replace manual change-penalty tuning with Bayesian optimisation.
            proposal = propose_control_vector(data_frame, change_penalty=self.auto_control_change_penalty)

        except Exception as fault:
            self._set_auto_control_status("waiting", f"No proposal - {fault}")
            return False

        current = [self.pulse_to_voltage(pin) for pin in self.pins[:CONTROL_PIN_COUNT]]
        review = self.autonomy_guard.review(proposal, current)

        if review.targets is None:
            self._set_auto_control_status("holding", f"Proposal rejected - {review.reason}")
            return False

        self.set_pin_voltages(review.targets)

        self._set_auto_control_status("running", f"Limited - {review.reason}" if review.action == "limit" else "")
        self.auto_control_status["last_update"] = time.time()

        return True

    def _auto_control_manager(self):
        """Background loop that runs auto control at its configured cadence."""

        # Scheduled against a deadline so the time a proposal takes comes out of the
        # period rather than being added on top of it.
        next_due = time.monotonic()

        while self.alive.is_set():

            next_due += self.auto_control_period_ms / 1000.0
            now = time.monotonic()

            # A step that overran its slot restarts the schedule from now instead of
            # firing a burst of back-to-back steps to catch up.
            if next_due < now:
                next_due = now

            if self._stop_event.wait(next_due - now):
                break

            try:
                self._auto_control_step()

            except Exception as fault:
                self._set_auto_control_status("error", str(fault))

    def _update_manager(self):
        """Background loop for periodic online model updates."""
        
        while self.alive.is_set():

            if self._stop_event.wait(self.online_update_period):
                break

            if not self.online_update_enabled or online_update is None:
                continue

            try:
                if str(self.sweep_status.get("state", "")).lower() == "running":
                    continue

            except Exception as fault:
                log.warning(f"WARNING: Sweep Status unretrievable - {fault}!")
                pass

            data_frame = self.get_training_data()

            if data_frame.empty:
                log.warning("WARNING: No hardware data available for updates (data frame is empty)!")
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
                updated, loss, r2 = online_update(window_data_frame, grad_clip_threshold=1.0)
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
            if self.offline or not self.serial or not self.serial.is_open:

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
                        for offset, volts in enumerate(raw_values, start = 1):

                            pwm_value = self.voltage_to_pulse(volts)
                            self.update_pin_values(offset, pwm_value)

                        log.info(f"[SIMULATED] TARGETS applied: {raw_values}")

                # A simulated board arms like a real one. A real board that has dropped
                # off the link is never reported as armed.
                elif command == "ARM" and self.force_offline:
                    if self.board_mode != "FAULT":
                        self.board_mode = "ARMED"
                    log.info(f"[SIMULATED] Board {self.board_mode}")

                elif command == "DISARM" and self.force_offline:
                    if self.board_mode != "FAULT":
                        self.board_mode = "SAFE"
                    for pin_index in range(1, 7):
                        self.update_pin_values(pin_index, 0)
                    log.info("[SIMULATED] Board disarmed, outputs zeroed")

                elif command_string.strip().upper() == "CLEAR FAULTS" and self.force_offline:
                    if self.board_mode == "FAULT":
                        self.board_mode = "SAFE"

                else:
                    log.info(f"[SIMULATED] Received command: {command_string}")
                return

            # The sweep, the dashboard callbacks and the keepalive all share one port. Each
            # command goes out with its CRC, and is noted so its reply can be checked off.
            word = _command_word(command_string)

            with self.command_lock:
                if word in COMMAND_REPLIES:
                    with self._pending_lock:
                        self._pending_replies.append((time.time(), word, COMMAND_REPLIES[word]))

                self.serial.write((frame_line(command_string.strip()) + "\n").encode("utf-8"))

            log.info(f"Command sent to board: {command_string}...")

        except Exception as fault:
            log.error(f"ERROR: Failed to send command '{command_string}' - {fault}!")

    def get_data(self) -> pd.DataFrame:
        """Returns a copy of the current rolling DataFrame."""

        with self.data_lock:
            data_frame = self.data_frame.copy()
            return data_frame

    def get_training_data(self) -> pd.DataFrame:
        """Returns the buffered samples that the model is allowed to learn from."""

        return self._admissible_for_training(self.get_data())

    def _admissible_for_training(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        """Drops the samples the model must not learn from.

        Simulated samples are only admissible when simulation was asked for. If the
        board drops out mid-run the simulator keeps the dashboard alive, but those
        samples must never reach the RNN as though they were diode readings.
        """

        if self.force_offline or data_frame.empty or "source" not in data_frame.columns:
            return data_frame

        return data_frame[data_frame["source"] != SIMULATED_SOURCE]

    def get_status(self) -> str:
        """Returns a connection/status string."""

        if self.force_offline:
            return "Simulating operation..."
        
        if self.serial and getattr(self.serial, "is_open", False):
            return f"Connected via {self.port}..."

        if self.offline:
            return "Disconnected - retrying connection..."

        return "Connecting..."

    def get_pin_value(self, i: int, value: int):
        """Clamp a requested pin value based on pin type (PWM pins 1..5 vs switch pin 6)."""

        if not (1 <= i <= 6):
            log.warning("WARNING: Pin index must be 1-6!")
            return None
        
        if i <= 5:
            return max(0, min(MAX_MODULATION_VALUE, int(value)))
        
        return 1 if int(value) else 0

    def set_pin_voltage(self, i: int, voltage: int):
        """Set a single pin by index (1..6) using raw PWM duty (pins 1..5) or 0/1 (pin 6)."""

        if i < 1 or i > 6:
            raise ValueError("ERROR: index must be 1-6!")
        
        if i <= 5:
            pin_voltage = max(0, min(MAX_MODULATION_VALUE, int(self.voltage_to_pulse(voltage))))

        else:
            pin_voltage = 1 if int(voltage) else 0

        self.send_command(f"PIN {int(i)} {int(pin_voltage)}")

    def set_pin_voltages(self, voltages):
        """Sets the first 5 control pins using given voltage targets"""

        if voltages is None:
            raise ValueError("ERROR: Missing voltage targets!")

        try:
            values = [float(voltage_output) for voltage_output in voltages]

        except Exception as fault:
            raise ValueError(f"ERROR: Invalid voltage target - {fault}!")

        if len(values) < CONTROL_PIN_COUNT:
            raise ValueError(f"ERROR: Invalid number of outputs requested, I need {CONTROL_PIN_COUNT} voltages but I only received {len(values)}!")

        target_voltages = [self.clamp_voltage(voltage_output) for voltage_output in values[:CONTROL_PIN_COUNT]]
        voltage_payload = "TARGETS " + " ".join(f"{voltage_output:.6f}" for voltage_output in target_voltages)

        self.send_command(voltage_payload)

    def set_pwm(self, channel: int, duty: int):
        """Sets PWM duty on control pins 1..5 using the set pin voltage function."""

        if channel < 1 or channel > 5:
            raise ValueError(f"ERROR: Pulse source must be a channel in range: 1-5. Current channel is: {channel}!")
        
        self.set_pin_voltage(channel, duty)

    def set_switch_timing(self, timing_input: float):
        """Sets the switch timing (microseconds between edges) given to the control board.

        The board takes whole microseconds only, so a fractional period is refused rather
        than truncated. One outside the shared bounds is clamped, with a warning.
        """

        try:
            switch_timing = float(timing_input)

        except Exception as fault:
            log.warning(f"WARNING: Switch timing not set: {fault}!")

            return

        if not switch_timing.is_integer():
            log.warning(f"WARNING: Switch timing not set: {switch_timing} us is not a whole number of microseconds!")

            return

        board_switch_timing = max(float(SWITCH_PERIOD_MIN_US), min(float(SWITCH_PERIOD_MAX_US), switch_timing))

        if board_switch_timing != switch_timing:
            log.warning(
                f"WARNING: Switch timing {switch_timing:.0f} us is outside {SWITCH_PERIOD_MIN_US}-{SWITCH_PERIOD_MAX_US} us, "
                f"clamped to {board_switch_timing:.0f} us!"
            )

        self.switch_timing = board_switch_timing

        if self.offline:
            try:
                log.info(f"[SIMULATED] Switch timing set to {board_switch_timing:.1f} us")

            except Exception as fault:
                log.warning(f"[SIMULATED] WARNING: Failed to log switch timing: {fault}!")
                pass

            return

        try:
            self.send_command(f"SWITCH_PERIOD_US {int(board_switch_timing)}")

        except Exception as fault:
            log.error(f"ERROR: Failed to send switch timing '{board_switch_timing}': {fault}!")

    def set_pin_by_name(self, name: str, value: int):
        """Sets a modulation value according to pin name/index."""

        i = self._name_to_pin_index(name)

        if not (1 <= i <= 6):
            raise ValueError("ERROR: Unknown pin index!")
        
        if i <= 5:
            value = max(0, min(MAX_MODULATION_VALUE, int(value)))

        else:
            value = 1 if int(value) else 0
        self.send_command(f"PIN {int(i)} {int(value)}")

    def get_pins(self):
        """Returns cached pin names/values and the last-update timestamp from the rolling dataframe."""

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
        """Stops background threads and closes the serial connection."""

        log.info("Backend thread disconnecting...")

        # A clean shutdown leaves the rig safe at once rather than after the failsafe timeout.
        if self.serial is not None and getattr(self.serial, "is_open", False):
            self.disarm()

        self.alive.clear()
        self._stop_event.set()
        self.disconnect()

        # Waits the workers out so a later start() cannot end up running two of each.
        for worker in self._workers():
            if worker is not threading.current_thread():
                worker.join(timeout=5.0)

        log.info("Backend thread disconnected...")

    # ------------------------------------------------------------------ #
    #                         Training sweep control                     #  
    # ------------------------------------------------------------------ #

    def _RNN_training_sweeps(self, minimum_voltage: float, maximum_voltage: float, voltage_step_size: float, hold_time: float, epochs: int, reference_voltages: int | None = None, factorial_levels: int | None = None, random_samples: int | None = None):
        """Internal worker for parameter sweeps + optional RNN training. Checks and generates the voltage sweeps, then iterates 
        through the combinations; while checking for cancellation and updating progress, sending all instructions then to the RNN."""

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

            voltage_range = max(0.0, maximum_voltage - minimum_voltage)

            try:
                voltage_step_size = float(voltage_step_size)

            except Exception as fault:
                log.warning(f"WARNING: Invalid voltage_step_size value - {fault}! Reverting to defaults...")
                voltage_step_size = 0.05

            if voltage_step_size <= 0:
                voltage_step_size = max(0.01, voltage_range / 25.0) if voltage_range > 0 else 0.05

            try:
                hold_time = max(0.01, float(hold_time))

            except Exception as fault:
                log.warning(f"WARNING: Invalid hold time - {fault}! Reverting to defaults...")
                hold_time = 0.05

            voltage_training_grid = np.arange(minimum_voltage, maximum_voltage + 1e-9, voltage_step_size, dtype=float)

            if voltage_training_grid.size == 0:
                log.warning("WARNING: voltage_training_grid size must be  non-zero! Using all available information...")
                voltage_training_grid = np.array([minimum_voltage], dtype=float)

            voltage_training_grid = np.clip(voltage_training_grid, minimum_voltage, maximum_voltage)

            try:
                num_reference_voltages = int(reference_voltages) if reference_voltages is not None else (3 if voltage_range > 0 else 1)

            except Exception as fault:
                log.warning(f"WARNING: Invalid baseline levels - {fault}! Reverting to defaults...")
                num_reference_voltages = 3 if voltage_range > 0 else 1
                
            num_reference_voltages = max(1, num_reference_voltages)
            
            reference_voltages = np.linspace(minimum_voltage, maximum_voltage, num=num_reference_voltages, dtype=float)
            reference_voltages = np.unique(np.round(reference_voltages, 6))

            if reference_voltages.size == 0:
                log.warning("WARNING: Reference voltages size must be non-zero! Using minimum voltages...")
                reference_voltages = np.array([minimum_voltage], dtype=float)

            try:
                factorial_level_count = int(factorial_levels) if factorial_levels is not None else (3 if voltage_range > 0 else 1)

            except Exception as fault:
                log.warning(f"WARNING: Invalid factorial levels - {fault}! Reverting to defaults...")
                factorial_level_count = 3 if voltage_range > 0 else 1

            factorial_level_count = max(1, factorial_level_count)
            factorial_levels = np.linspace(minimum_voltage, maximum_voltage, num=factorial_level_count, dtype=float)
            factorial_levels = np.clip(factorial_levels, minimum_voltage, maximum_voltage)

            if random_samples is None:
                log.warning("WARNING: Random samples not specified! Using default value...")
                random_sample_count = max(int(len(voltage_training_grid) * CONTROL_PIN_COUNT), 20)

            else:
                try:
                    random_sample_count = int(random_samples)

                except Exception as fault:
                    log.warning(f"WARNING: Invalid random sample - {fault}! Reverting to defaults...")
                    random_sample_count = 0

                if random_sample_count <= 0:
                    log.warning("WARNING: Random sample count must be positive! Using default value...")
                    random_sample_count = max(int(len(voltage_training_grid) * CONTROL_PIN_COUNT), 20)

            stage1_steps = int(len(reference_voltages) * CONTROL_PIN_COUNT * len(voltage_training_grid))
            stage2a_steps = int(len(factorial_levels) ** CONTROL_PIN_COUNT)

            total_steps = stage1_steps + stage2a_steps + random_sample_count

            if total_steps <= 0:
                log.warning("WARNING: Total number of sweep steps must be positive! Using 1...")
                total_steps = 1

            random_number_generator = np.random.default_rng()
            completion_indicator = 0

            class SweepAbort(Exception):
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
                completion_indicator += 1

                self.sweep_status["progress"] = min(1.0, completion_indicator / total_steps)
                self.sweep_status["message"] = stage_label

            self._begin_sweep_capture()

            try:
                for reference in reference_voltages:

                    base_vector = [reference] * CONTROL_PIN_COUNT

                    for pin_index in range(CONTROL_PIN_COUNT):
                        for voltage_interval in voltage_training_grid:

                            voltage_targets = list(base_vector)
                            voltage_targets[pin_index] = float(voltage_interval)

                            run_step(voltage_targets,f"Sweep 1: pin {pin_index + 1} sweeping at {reference:.2f} V",)

                combo_total = max(1, stage2a_steps)
                combo_index = 0

                for combination_sweep_index in itertools.product(factorial_levels, repeat=CONTROL_PIN_COUNT):

                    combo_index += 1
                    run_step(combination_sweep_index, f"Sweep 2: factorial sweep for the RNN ({combo_index}/{combo_total})")

                for sample_index in range(1, random_sample_count + 1):

                    random_targets = random_number_generator.uniform(minimum_voltage, maximum_voltage, CONTROL_PIN_COUNT)
                    run_step(random_targets.tolist(), f"Stage 3: random sampling for the RNN ({sample_index}/{random_sample_count})",)

            except SweepAbort:
                log.warning("WARNING: Sweep aborted by user!")
                self.sweep_status.update({"state": "aborted", "message": "Sweep aborted by user..."})
                return

            finally:
                sweep_capture = self._end_sweep_capture()

            # Trains on the whole sweep, not whatever the rolling buffer still happens to hold.
            data_frame = self._admissible_for_training(sweep_capture)

            try:
                if data_frame.empty:

                    self.sweep_status.update({
                        "state": "failed",
                        "message": "ERROR: Sweep Failed! No hardware data collected during sweep attempt...",
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
                            "source",
                        ]

                        available_columns = [column for column in ordered_columns if column in data_frame.columns]

                        sweep_dataset = data_frame.loc[:, available_columns].rename(
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
                        sweep_dataset.to_csv(file_path, index=False)

                        log.info(f"Sweep dataset successfully saved to {file_path}...")

                    except Exception as fault:
                        log.error(f"ERROR: Failed to save sweep dataset - {fault}! Please initalise the sweep again, or continue without saving...")

                if train_model is not None:
                    try:
                        metrics = train_model(data_frame, number_of_epochs=int(max(1, epochs)))
                        current_time = time.time()

                        self.last_training_time_stamp = current_time
                        self.last_update_time_stamp = current_time

                        try:
                            if isinstance(metrics, dict):

                                ML_METRICS.record_training_update(current_time, loss = metrics.get("loss"), r2 = metrics.get("r2"), source = "sweep")

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
        voltage_step_size: float = 0.05,
        step_linger_time: float = 0.05,
        epochs: int = 10,
        reference_voltages: int | None = None,
        factorial_levels: int | None = None,
        random_samples: int | None = None,
    ):
        """Initialises a training sweep."""

        if self.sweep_thread and self.sweep_thread.is_alive():
            log.info("Sweep already running! Please wait before attempting to start another training sweep...")
            return False

        # A sweep drives every output, so it needs the board armed first like any operator would.
        if not self.is_armed():
            self.sweep_status = {"state": "idle", "progress": 0.0, "message": f"Board is not armed ({self.board_mode})"}
            log.warning("WARNING: Sweep refused, the board is not armed!")
            return False

        self.sweep_cancel.clear()
        self.sweep_status = {"state": "queued", "progress": 0.0, "message": ""}

        self.sweep_thread = threading.Thread(target=self._RNN_training_sweeps,args=(min_voltage, 
                                                                                    max_voltage, 
                                                                                    voltage_step_size, 
                                                                                    step_linger_time, 
                                                                                    epochs, 
                                                                                    reference_voltages, 
                                                                                    factorial_levels, 
                                                                                    random_samples), 
                                                                                    daemon=True)
        self.sweep_thread.start()

        return True

    def get_sweep_status(self) -> dict:
        """Returns the latest sweep status message."""

        return dict(self.sweep_status)

    def stop_training_sweep(self):
        """Requests cancellation of an in-progress sweep."""

        self.sweep_cancel.set()


# ------------------------------------------------------------------------- #
#                               Board Access                                #    
# ------------------------------------------------------------------------- #

# OFFLINE environment variable can be used to force simulated mode.
status = OFFLINE

Back_End_Controller = SerialBackend(status = status)
Data_Reciever = Back_End_Controller


def start_backend():
    """Starts the shared backend. Entry points call this - importing the module never does."""

    return Back_End_Controller.start()


get_data = Back_End_Controller.get_data
get_training_data = Back_End_Controller.get_training_data
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
set_optimiser_type = getattr(Back_End_Controller, "set_optimiser_type", None)
get_online_update_config = getattr(Back_End_Controller, "get_online_update_config", None)
set_buffer_samples = getattr(Back_End_Controller, "set_buffer_samples", None)
set_auto_control = Back_End_Controller.set_auto_control
get_auto_control = Back_End_Controller.get_auto_control
get_buffer_samples = getattr(Back_End_Controller, "get_buffer_samples", None)

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
        "optimiser_type": None,
        "sweep_state": None,
        "data_buffer_samples": None,
    }

    current_time = time.time()

    try:
        ts = getattr(Back_End_Controller, "last_training_time_stamp", None)

        if ts is not None:
            status_snapshot["last_train_ago_sec"] = max(0.0, float(current_time - float(ts)))

    except Exception as fault:
        log.warning(f"WARNING: Couldnot access the last training time voltage_step_size: {fault}!")
        pass
    
    try:
        online_ts = getattr(Back_End_Controller, "last_update_time_stamp", None)

        if online_ts is not None:
            status_snapshot["last_online_update_ago_sec"] = max(0.0, float(current_time - float(online_ts)))
   
    except Exception as fault:
        log.warning(f"WARNING: Couldnot access the last online update time voltage_step_size: {fault}!")
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
        status_snapshot["optimiser_type"] = _get_rnn_optimiser_type()

    except Exception as fault:
        log.warning(f"WARNING: Couldnot access the optimiser type: {fault}!")
        status_snapshot["optimiser_type"] = None
   
    try:
        status_snapshot["sweep_state"] = str(getattr(Back_End_Controller, "sweep_status", {}).get("state", "idle"))
    
    except Exception as fault:
        log.warning(f"WARNING: Couldnot access the sweep state: {fault}!")
        status_snapshot["sweep_state"] = None

    try:
        status_snapshot["data_buffer_samples"] = int(getattr(Back_End_Controller, "max_buffer_samples", None))
    except Exception as fault:
        log.warning(f"WARNING: Couldnot access the buffer sample size: {fault}!")
        status_snapshot["data_buffer_samples"] = None
    
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
    start_backend()

    try:
        while True:
            time.sleep(1)
            data_frame = Back_End_Controller.get_data()
            
            if not data_frame.empty:
                log.info(f"Latest: {data_frame['voltage'].iloc[-1]:.3f} V ({len(data_frame)} samples)")

    except KeyboardInterrupt:
        Back_End_Controller.stop()
        log.info("Backend stopped...")
