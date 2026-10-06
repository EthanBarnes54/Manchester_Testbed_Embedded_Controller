"""Test doubles and data shared across the suite."""

import queue
import threading
import time

import numpy as np
import pandas as pd


class FakePort:
    """Plays the board's side of the serial link. Serves queued lines, falls back to a
    steady MEASURED stream while streaming is on, records everything written, and answers
    the safety and health commands (ARM, DISARM, FAULTS, CLEAR FAULTS, HEALTH, SELFTEST)
    the way the firmware does."""

    def __init__(self):
        self.lines = queue.Queue()
        self.written = []
        self.opened = []
        self.streaming = threading.Event()
        self.streaming.set()
        self.line_interval = 0.01
        self.mode = "SAFE"
        self.selftest = "PASS"

    def commands(self, prefix=""):
        return [command for _, command in self.written if command.startswith(prefix)]

    def faults_line(self):
        return (f"FAULTS mode={self.mode} active=none latched=none history=none counts=none "
                "boots=1 unexpected_resets=0 last_reset=POWERON")

    def respond(self, command):
        word = command.strip().upper()

        if word == "ARM":
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

                try:
                    return (port.lines.get_nowait() + "\n").encode()
                except queue.Empty:
                    return b"MEASURED 1.23456 V\n" if port.streaming.is_set() else b""

            def write(self, data):
                command = data.decode().strip()
                port.written.append((time.time(), command))
                port.respond(command)

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
    """Synthetic diode data with each pin vector held for several samples.

    Independently varying pins make the next voltage unpredictable, so any model result
    on that kind of data is meaningless. Holding settings is what a real sweep does."""

    rng = np.random.default_rng(seed)
    rows, timestamp = [], 0.0

    for _ in range(holds):
        pins = rng.integers(0, 1024, 5)

        for _ in range(samples_per_hold):
            voltage = 0.5 + pins.mean() / 1023 * 2 + rng.normal(0, 0.01)
            rows.append({"timestamp": timestamp, "voltage": voltage, **{f"pin_{i + 1}": int(pins[i]) for i in range(5)}})
            timestamp += 0.05

    return pd.DataFrame(rows)
