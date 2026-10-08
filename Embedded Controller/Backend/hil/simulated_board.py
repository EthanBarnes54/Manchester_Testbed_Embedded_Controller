"""A simulated board at the serial protocol level, for checking the rig tests themselves.

TESTBED_HIL_PORT=sim runs the hardware-in-the-loop suite against this instead of a board,
so a mistake in a test (a wrong reply string, a timing assumption) shows up before anyone
is standing at the rig. It plays the firmware's documented behaviour: CRC checking, the
command set and its refusals, the 5 s failsafe, the 20 Hz numbered readings, the 256-byte
command buffer and a reset on the RTS line. It is a model of the protocol, not of the
chip: passing against it says nothing about the hardware.

TESTBED_SIM_FAULT makes it misbehave in one named way (see FAULTS), so the test suite can
show each rig test fails against a board that gets that one thing wrong.
"""

import math
import os
import queue
import threading
import time

from rig import frame, unframe

COMMAND_BUFFER_LIMIT = 256
COMMAND_TIMEOUT_S = 5.0
MEASUREMENT_INTERVAL_S = 0.05
SWITCH_PERIOD_MIN_US, SWITCH_PERIOD_MAX_US = 1, 2_000_000
PIN_NAMES = {"squeeze_plate": 1, "ion_source": 2, "wein_filter": 3, "cone_1": 4, "cone_2": 5, "switch_logic": 6}

FAULTS = {
    "ignores_crc": "acts on commands whose CRC does not match",
    "late_failsafe": "trips the failsafe at 6 s instead of 5 s",
    "energises_while_safe": "accepts output commands while disarmed",
    "drops_readings": "loses one reading in every 50",
    "uncounted_overflow": "discards an over-long line without counting it",
    "blocking_serial": "has no serial transmit buffer, so a pass that writes a lot overruns",
}

# Without a transmit buffer a write waits for the UART's 128-byte FIFO, which drains 11.52
# characters a millisecond at 115200 baud: more than this in one pass takes over 20 ms.
UNBUFFERED_PASS_LIMIT = 128 + 20 * 11.52


class SimulatedBoard:
    """Stands in for serial.Serial: the attributes and methods Rig uses, nothing more."""

    def __init__(self):
        self.port, self.baudrate, self.timeout = "sim", 115200, 0.05
        self.dtr = False
        self._rts = False
        self.boots = 0
        self.fault = os.environ.get("TESTBED_SIM_FAULT", "")
        assert self.fault in ("", *FAULTS), f"unknown TESTBED_SIM_FAULT {self.fault!r}"
        self._outgoing = queue.Queue()
        self._incoming = bytearray()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._boot()

    # ---------------------------------------------------------- serial.Serial

    @property
    def rts(self):
        return self._rts

    @rts.setter
    def rts(self, asserted):
        # The auto-reset circuit holds EN low while RTS is asserted; releasing it boots the chip.
        if self._rts and not asserted:
            with self._lock:
                self._boot()
        self._rts = asserted

    def open(self):
        self._thread = threading.Thread(target=self._run, name="simulated-board", daemon=True)
        self._thread.start()

    def close(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def write(self, data: bytes):
        with self._lock:
            self._incoming += data
        return len(data)

    def flush(self):
        pass

    def readline(self) -> bytes:
        try:
            return (self._outgoing.get(timeout=self.timeout) + "\n").encode("utf-8")
        except queue.Empty:
            return b""

    # ------------------------------------------------------------- firmware

    def _boot(self):
        self.boots += 1
        self.started = time.monotonic()
        self.mode, self.switch_us, self.targets = "SAFE", 0, [0.0] * 5
        self.active, self.history = set(), set()
        self.sequence, self.next_reading = 0, self.started + MEASUREMENT_INTERVAL_S
        self.buffer, self.rx_overflows, self.bad_checksums, self.overruns = "", 0, 0, 0
        self._pass_characters = 0
        self.last_command, self.command_seen, self.failsafe = 0.0, False, False
        self._incoming.clear()

    def _send(self, text):
        line = frame(text)
        self._pass_characters += len(line) + 2
        self._outgoing.put(line)

    def _millis(self):
        return int((time.monotonic() - self.started) * 1000)

    def _run(self):
        while not self._stop.wait(0.001):
            with self._lock:
                self._pass()

    def _pass(self):
        now = time.monotonic()
        incoming, self._incoming = bytes(self._incoming), bytearray()
        self._pass_characters = 0

        for byte in incoming:
            character = chr(byte)
            if character == "\r":
                continue
            if character == "\n":
                self._handle(self.buffer.strip())
                self.buffer = ""
            elif len(self.buffer) < COMMAND_BUFFER_LIMIT:
                self.buffer += character
            else:
                self.buffer = ""
                self.rx_overflows += self.fault != "uncounted_overflow"

        if now >= self.next_reading:
            self.sequence += 1 + (self.fault == "drops_readings" and self.sequence % 50 == 49)
            self.next_reading += MEASUREMENT_INTERVAL_S
            self._send(f"MEASURED 1.23456 V seq={self.sequence} t_ms={self._millis()}")

        timeout = COMMAND_TIMEOUT_S + (self.fault == "late_failsafe")
        overdue = self.command_seen and now - self.last_command >= timeout

        if overdue and not self.failsafe:
            self.failsafe = True
            self._make_safe()
            self._send("FAILSAFE outputs zeroed, no command received from host")
            self.active.add("HOST_TIMEOUT")
            self.history.add("HOST_TIMEOUT")
        elif not overdue and self.failsafe:
            self.failsafe = False
            self.active.discard("HOST_TIMEOUT")
            self._send("FAILSAFE cleared, host link restored")

        if self.fault == "blocking_serial" and self._pass_characters > UNBUFFERED_PASS_LIMIT:
            self.overruns += 1

    def _make_safe(self):
        self.switch_us, self.targets = 0, [0.0] * 5
        if self.mode == "ARMED":
            self.mode = "SAFE"

    def _handle(self, line):
        text, verdict = unframe(line)

        if verdict == "invalid" and self.fault != "ignores_crc":
            self.bad_checksums += 1
            self._send("ERROR: Bad checksum!")
            return

        command = text.strip()
        if not command:
            return

        self.last_command, self.command_seen = time.monotonic(), True
        upper = command.upper()
        armed = self.mode == "ARMED" or self.fault == "energises_while_safe"

        if upper == "PING":
            self._send("OK")
        elif upper == "VERSION":
            self._send("VERSION firmware=simulated protocol=2 build=simulated gate_loopback=1 setpoint_readback=0")
        elif upper == "ARM":
            if self.mode == "FAULT":
                self._send("ERROR: Cannot arm - a critical fault is latched, send CLEAR FAULTS!")
            else:
                self.mode = "ARMED"
                self._send("ACK ARM")
        elif upper == "DISARM":
            self._make_safe()
            self._send("ACK DISARM")
        elif upper == "FAULTS":
            listed = lambda faults: ",".join(sorted(faults)) or "none"  # noqa: E731
            self._send(f"FAULTS mode={self.mode} active={listed(self.active)} latched=none history={listed(self.history)} "
                       f"counts=none boots={self.boots} unexpected_resets=0 last_reset=POWERON")
        elif upper == "SELFTEST":
            self._send(f"SELFTEST PASS clocks=ok switch=ok adc=ok memory=ok gate=ok readback=off mode={self.mode}")
        elif upper == "HEALTH":
            switching = armed and self.switch_us > 0
            self._send(f"HEALTH mode={self.mode} uptime_ms={self._millis()} loop_max_us=900 loop_peak_us=2400 "
                       f"loop_budget_us=20000 overruns={self.overruns} heap_free=200000 heap_min=190000 stack_free=5000 adc=ok "
                       f"adc_conversions={self.sequence} adc_timeouts=0 rx_overflows={self.rx_overflows} "
                       f"bad_checksums={self.bad_checksums} gate=ok gate_edges={1000 if switching else 0} readback=off")
        elif command.startswith("TARGETS"):
            values = command.split()[1:]
            try:
                volts = [float(value) for value in values]
            except ValueError:
                volts = []
            # As the firmware does: each voltage becomes a 10-bit duty, and only a non-zero
            # duty counts as energising. Each channel is acknowledged, then the command.
            duties = [int(min(max(v, 0.0), 3.3) / 3.3 * 1023 + 0.5) for v in volts if math.isfinite(v)]
            if len(volts) != 5 or len(duties) != 5:
                self._send("ERROR: TARGETS requires five voltages!")
            elif not armed and any(duties):
                self._send("ERROR: Not armed!")
            else:
                self.targets = volts
                for channel, duty in enumerate(duties, start=1):
                    self._send(f"ACK PIN {channel} {duty}")
                self._send("ACK TARGETS")
        elif command.startswith("PIN"):
            parts = command.split()
            if len(parts) != 3:
                self._send("ERROR: Invalid PIN syntax!")
                return
            index = PIN_NAMES.get(parts[1].lower()) or (int(parts[1]) if parts[1].isdigit() else 0)
            value = int(parts[2]) if parts[2].lstrip("-").isdigit() else 0
            if not armed and value > 0:
                self._send("ERROR: Not armed!")
            elif not 1 <= index <= 6:
                self._send("ERROR: PIN index out of range!")
            else:
                self._send(f"ACK PIN {index} {value}")
        elif command.startswith("SWITCH_PERIOD_US"):
            parts = command.split()
            period = int(parts[1]) if len(parts) == 2 and parts[1].lstrip("-").isdigit() else 0
            if len(parts) != 2:
                self._send("ERROR: Invalid switch time!")
            elif period <= 0:
                self.switch_us = 0
                self._send("ACK SWITCH_PERIOD_US 0 (disabled)")
            elif not armed:
                self._send("ERROR: Not armed!")
            elif not SWITCH_PERIOD_MIN_US <= period <= SWITCH_PERIOD_MAX_US:
                self._send("ERROR: Switch time out of bounds!")
            else:
                self.switch_us = period
                self._send(f"ACK SWITCH_PERIOD_US {period}")
        else:
            self._send("ERROR: Unknown command received!")
