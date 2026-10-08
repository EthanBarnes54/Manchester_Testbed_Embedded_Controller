import binascii
import queue
import threading
import time

import numpy as np
import pandas as pd


def crc_suffix(text):
    """The link's CRC as the firmware appends it (CRC-16/CCITT-FALSE, "*XXXX")."""

    return f"*{binascii.crc_hqx(text.encode(), 0xFFFF):04X}"


class FakePort:
    """Plays the board's side of the serial link, speaking protocol 2 like the firmware."""

    def __init__(self):
        self.lines = queue.Queue()
        self.written = []
        self.raw_written = []
        self.opened = []
        self.streaming = threading.Event()
        self.streaming.set()
        self.line_interval = 0.01
        self.mode = "SAFE"
        self.selftest = "PASS"
        self.protocol = 2
        self.sequence = 0

    def commands(self, prefix=""):
        return [command for _, command in self.written if command.startswith(prefix)]

    def put_raw(self, line):
        self.lines.put(_Raw(line))

    def serve(self):
        """The next line the board sends, CRC and all."""

        try:
            line = self.lines.get_nowait()
        except queue.Empty:
            if not self.streaming.is_set():
                return ""
            self.sequence += 1
            line = f"MEASURED 1.23456 V seq={self.sequence} t_ms={50 * self.sequence}"

        if isinstance(line, _Raw):
            return str(line)

        return line + crc_suffix(line)

    def faults_line(self):
        return (f"FAULTS mode={self.mode} active=none latched=none history=none counts=none "
                "boots=1 unexpected_resets=0 last_reset=POWERON")

    def respond(self, command):
        word = command.strip().upper()

        if word == "VERSION":
            self.lines.put(f"VERSION firmware=fake protocol={self.protocol} build=test gate_loopback=0 setpoint_readback=0")

        elif word == "ARM":
            if self.mode == "FAULT":
                self.lines.put("ERROR: Cannot arm - a critical fault is latched, send CLEAR FAULTS!")
            else:
                self.mode = "ARMED"
                self.lines.put("ACK ARM")

        elif word == "DISARM":
            if self.mode != "FAULT":
                self.mode = "SAFE"
            self.lines.put("ACK DISARM")

        elif word == "FAULTS":
            self.lines.put(self.faults_line())

        elif word == "CLEAR FAULTS":
            if self.mode == "FAULT":
                self.mode = "SAFE"
            self.lines.put(f"ACK CLEAR FAULTS mode={self.mode}")

        elif word == "HEALTH":
            self.lines.put(f"HEALTH mode={self.mode} uptime_ms=1000 loop_max_us=1100 loop_peak_us=2400 "
                           "loop_budget_us=20000 overruns=0 heap_free=200000 heap_min=190000 stack_free=5000 "
                           "adc=ok adc_conversions=20 adc_timeouts=0 rx_overflows=0 gate=off gate_edges=0 readback=off")

        elif word == "SELFTEST":
            self.lines.put(f"SELFTEST {self.selftest} clocks=ok switch=ok adc=ok memory=ok gate=off readback=off mode={self.mode}")

    def make_serial_class(self):
        port = self

        class FakeSerial:
            def __init__(self, name, baud, timeout=1):
                port.opened.append(name)
                self.is_open = True

            def readline(self):
                time.sleep(port.line_interval)
                line = port.serve()
                return (line + "\n").encode() if line else b""

            def write(self, data):
                raw = data.decode().strip()
                text, valid = raw, None

                if len(raw) >= 5 and raw[-5] == "*":
                    text, valid = raw[:-5], crc_suffix(raw[:-5]) == raw[-5:]

                port.raw_written.append((raw, valid))

                if valid is False:
                    port.lines.put("ERROR: Bad checksum!")
                    return

                port.written.append((time.time(), text))
                port.respond(text)

            def close(self):
                self.is_open = False

        return FakeSerial


def wait_for(condition, timeout=5.0, interval=0.02):
    """Polls until condition() is truthy or the timeout passes. Returns the last result."""

    deadline = time.time() + timeout
    result = condition()

    while not result and time.time() < deadline:
        time.sleep(interval)
        result = condition()

    return result


def sweep_shaped_frame(holds=40, samples_per_hold=8, seed=0):
    """Synthetic diode data with each pin vector held for several samples."""

    rng = np.random.default_rng(seed)
    rows, timestamp = [], 0.0

    for _ in range(holds):
        pins = rng.integers(0, 1024, 5)

        for _ in range(samples_per_hold):
            voltage = 0.5 + pins.mean() / 1023 * 2 + rng.normal(0, 0.01)
            rows.append({"timestamp": timestamp, "voltage": voltage, **{f"pin_{i + 1}": int(pins[i]) for i in range(5)}})
            timestamp += 0.05

    return pd.DataFrame(rows)


class _Raw(str):
    """A line FakePort serves exactly as given, without adding a CRC."""
