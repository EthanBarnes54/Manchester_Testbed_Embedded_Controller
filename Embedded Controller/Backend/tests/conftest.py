"""Shared fixtures. No test can reach real hardware: the module-level backend runs in
simulation, and pyserial is replaced before anything imports it."""

import os
import time

import pytest
import serial

from helpers import FakePort

# Set before python_Backend is first imported, so its shared backend is simulated.
os.environ["OFFLINE"] = "1"


class _NoHardware:
    """Default stand-in for serial.Serial. Any test that forgets to fake the port fails to connect."""

    def __init__(self, *args, **kwargs):
        raise serial.SerialException("tests must never open a real serial port")


serial.Serial = _NoHardware


@pytest.fixture
def fake_port(monkeypatch):
    port = FakePort()
    monkeypatch.setattr(serial, "Serial", port.make_serial_class())
    return port


@pytest.fixture
def backend_module():
    import python_Backend

    return python_Backend


@pytest.fixture
def live_backend(backend_module, fake_port):
    """A started backend talking to the fake port, with online learning off so it cannot
    interfere with timing, and always stopped afterwards."""

    backend = backend_module.SerialBackend(port="FAKE0", status=False)
    backend.online_update_enabled = False
    backend.start()

    deadline = time.time() + 5
    while backend.get_data().empty and time.time() < deadline:
        time.sleep(0.02)

    yield backend
    backend.stop()


@pytest.fixture
def shared_backend(backend_module, monkeypatch):
    """The module-level backend the dashboard drives, with commands captured instead of
    sent and its settings put back afterwards."""

    backend = backend_module.Back_End_Controller
    sent = []
    monkeypatch.setattr(backend, "send_command", sent.append)
    backend.sent = sent

    yield backend

    del backend.sent
    backend.set_buffer_samples(backend_module.DEFAULT_DATA_BUFFER_SAMPLES)
    backend.set_auto_control(
        enabled=False,
        period_ms=backend_module.AUTO_CONTROL_DEFAULT_PERIOD_MS,
        change_penalty=backend_module.AUTO_CONTROL_DEFAULT_CHANGE_PENALTY,
    )
    backend.sweep_status = {"state": "idle", "progress": 0.0, "message": ""}
    backend.set_save_dataset_enabled(False)
