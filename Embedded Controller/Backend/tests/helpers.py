"""Test doubles and data shared across the suite."""

import queue
import threading
import time

import numpy as np
import pandas as pd


class FakePort:
    """Plays the board's side of the serial link. Serves queued lines, falls back to a
    steady MEASURED stream while streaming is on, and records everything written."""

    def __init__(self):
        self.lines = queue.Queue()
        self.written = []
        self.opened = []
        self.streaming = threading.Event()
        self.streaming.set()
        self.line_interval = 0.01

    def commands(self, prefix=""):
        return [command for _, command in self.written if command.startswith(prefix)]

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
                port.written.append((time.time(), data.decode().strip()))

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
