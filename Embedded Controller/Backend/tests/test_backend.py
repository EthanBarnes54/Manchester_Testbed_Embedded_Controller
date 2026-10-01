"""The serial backend, driven through a fake port so no board is needed."""

import threading
import time

import pandas as pd
import pytest

from helpers import wait_for


def backend_threads():
    return sorted(thread.name for thread in threading.enumerate() if thread.name.startswith("backend-"))


# ----------------------------------------------------------------------------
#                           Configuration from the environment
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, expected",
    [(None, False), ("", False), ("0", False), ("false", False), ("FALSE", False), ("Off", False), ("no", False),
     ("1", True), ("true", True), ("YES", True)],
)
def test_offline_flag_is_case_folded(backend_module, monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("OFFLINE", raising=False)
    else:
        monkeypatch.setenv("OFFLINE", value)

    assert backend_module._env_flag("OFFLINE") is expected


@pytest.mark.parametrize("value, expected", [(None, "COM4"), ("", "COM4"), ("  ", "COM4"), ("/dev/ttyUSB0", "/dev/ttyUSB0")])
def test_serial_port_comes_from_the_environment(backend_module, monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("TESTBED_SERIAL_PORT", raising=False)
    else:
        monkeypatch.setenv("TESTBED_SERIAL_PORT", value)

    assert backend_module._configured_serial_port() == expected


# ----------------------------------------------------------------------------
#                                Lifecycle
# ----------------------------------------------------------------------------


def test_importing_the_backend_touches_no_hardware(backend_module):
    shared = backend_module.Back_End_Controller

    assert not shared.alive.is_set()
    assert shared.thread is None
    assert backend_threads() == []


def test_start_opens_the_port_and_stop_joins_every_worker(backend_module, fake_port):
    backend = backend_module.SerialBackend(port="FAKE7", status=False)
    backend.online_update_enabled = False

    assert fake_port.opened == []
    backend.start()
    backend.start()  # a second start must not double up

    assert wait_for(lambda: fake_port.opened == ["FAKE7"])
    assert backend_threads() == ["backend-auto-control", "backend-keepalive", "backend-online-update", "backend-serial"]

    started = time.time()
    backend.stop()

    assert time.time() - started < 3.0
    assert backend_threads() == []


# ----------------------------------------------------------------------------
#                            Reading from the board
# ----------------------------------------------------------------------------


def test_measurements_are_buffered_as_hardware_readings(live_backend):
    latest = live_backend.get_data().iloc[-1]

    assert latest["voltage"] == pytest.approx(1.23456)
    assert latest["source"] == "hardware"


def test_pin_reports_and_acknowledgements_update_the_cached_pins(live_backend, fake_port):
    fake_port.lines.put("PINS squeeze_plate=10 ion_source=20 wein_filter=30 cone_1=40 cone_2=50 switch_logic=1")
    assert wait_for(lambda: live_backend.get_pins()["values"] == [10, 20, 30, 40, 50, 1])

    fake_port.lines.put("ACK PIN 3 777")
    assert wait_for(lambda: live_backend.get_pins()["values"][2] == 777)


@pytest.fixture
def fast_keepalive(backend_module, monkeypatch):
    monkeypatch.setattr(backend_module, "KEEPALIVE_INTERVAL_SEC", 0.1)


def test_keepalive_pings_a_quiet_link(fast_keepalive, live_backend, fake_port):
    assert wait_for(lambda: len(fake_port.commands("PING")) >= 2)


def test_commands_are_newline_terminated_on_the_wire(live_backend, fake_port):
    live_backend.set_pin_voltages([1.0, 0.5, 0.0, 3.3, 9.0])

    assert fake_port.commands("TARGETS") == ["TARGETS 1.000000 0.500000 0.000000 3.300000 3.300000"]


def test_switch_timing_is_clamped_to_the_shared_bounds(backend_module, live_backend, fake_port):
    live_backend.set_switch_timing(1)
    live_backend.set_switch_timing(10 ** 9)

    assert fake_port.commands("SWITCH_PERIOD_US") == [
        f"SWITCH_PERIOD_US {backend_module.SWITCH_PERIOD_MIN_US}",
        f"SWITCH_PERIOD_US {backend_module.SWITCH_PERIOD_MAX_US}",
    ]


# ----------------------------------------------------------------------------
#                                Provenance
# ----------------------------------------------------------------------------


def test_simulated_samples_never_reach_training_on_a_real_run(backend_module):
    backend = backend_module.SerialBackend(port="FAKE0", status=False)
    backend._append_measurement(1.0, 1.0, "MEASURED 1.0 V")
    backend._append_measurement(2.0, 2.0, "MEASURED 2.0 SIMULATED", source=backend_module.SIMULATED_SOURCE)

    assert backend.get_training_data()["voltage"].tolist() == [1.0]


def test_simulated_samples_are_admissible_when_simulation_was_asked_for(backend_module):
    backend = backend_module.SerialBackend(port="FAKE0", status=True)
    backend._append_measurement(2.0, 2.0, "MEASURED 2.0 SIMULATED", source=backend_module.SIMULATED_SOURCE)

    assert len(backend.get_training_data()) == 1


# ----------------------------------------------------------------------------
#                                 Sweeps
# ----------------------------------------------------------------------------


def run_sweep(backend, **overrides):
    settings = dict(min_voltage=0.0, max_voltage=3.3, voltage_step_size=0.5, step_linger_time=0.02, epochs=1,
                    reference_voltages=1, factorial_levels=2, random_samples=10)
    settings.update(overrides)
    assert backend.start_training_sweep(**settings)
    backend.sweep_thread.join(timeout=60)


def test_a_sweep_trains_on_everything_it_recorded_not_just_the_buffer(backend_module, live_backend, monkeypatch):
    trained_on = []
    monkeypatch.setattr(backend_module, "train_model", lambda frame, number_of_epochs=1: trained_on.append(frame) or {"loss": 0.0, "r2": 0.0})

    live_backend.set_buffer_samples(backend_module.DATA_BUFFER_MIN_SAMPLES)
    run_sweep(live_backend)

    assert live_backend.get_sweep_status()["state"] == "completed"
    assert len(trained_on[0]) > backend_module.DATA_BUFFER_MIN_SAMPLES
    assert set(trained_on[0]["source"]) == {"hardware"}


def test_an_aborted_sweep_closes_its_capture_and_skips_training(backend_module, live_backend, monkeypatch):
    trained = []
    monkeypatch.setattr(backend_module, "train_model", lambda *args, **kwargs: trained.append(1))

    assert live_backend.start_training_sweep(voltage_step_size=0.1, step_linger_time=0.05, epochs=1)
    time.sleep(0.3)
    live_backend.stop_training_sweep()
    live_backend.sweep_thread.join(timeout=10)

    assert live_backend.get_sweep_status()["state"] == "aborted"
    assert live_backend._sweep_capture is None
    assert trained == []


# ----------------------------------------------------------------------------
#                               Auto control
# ----------------------------------------------------------------------------


@pytest.fixture
def stub_proposer(backend_module, monkeypatch):
    proposals = []

    def propose(frame, change_penalty=0.1):
        proposals.append(change_penalty)
        return [1.0, 1.0, 1.0, 1.0, 1.0]

    monkeypatch.setattr(backend_module, "propose_control_vector", propose)
    return proposals


def test_auto_control_settings_are_clamped(backend_module):
    backend = backend_module.SerialBackend(port="FAKE0", status=False)

    assert backend.set_auto_control(period_ms=1)["period_ms"] == backend_module.AUTO_CONTROL_MIN_PERIOD_MS
    assert backend.set_auto_control(period_ms=10 ** 9)["period_ms"] == backend_module.AUTO_CONTROL_MAX_PERIOD_MS
    assert backend.set_auto_control(change_penalty=-5)["change_penalty"] == 0.0


def test_auto_control_acts_only_when_enabled_fresh_and_not_sweeping(backend_module, stub_proposer, fake_port):
    backend = backend_module.SerialBackend(port="FAKE0", status=False)
    backend.connect()

    assert backend._auto_control_step() is False
    assert backend.get_auto_control()["state"] == "off"

    backend.set_auto_control(enabled=True, change_penalty=0.3)
    assert backend._auto_control_step() is False
    assert backend.get_auto_control()["state"] == "waiting"  # nothing read yet

    backend._append_measurement(time.time() - 10, 1.0, "MEASURED 1.0 V")
    assert backend._auto_control_step() is False
    assert "old" in backend.get_auto_control()["message"]  # stale reading

    backend._append_measurement(time.time(), 1.0, "MEASURED 1.0 V")
    backend.sweep_status = {"state": "running"}
    assert backend._auto_control_step() is False
    assert backend.get_auto_control()["state"] == "paused"

    backend.sweep_status = {"state": "idle"}
    assert backend._auto_control_step() is True
    assert stub_proposer == [0.3]
    assert fake_port.commands("TARGETS") == ["TARGETS 1.000000 1.000000 1.000000 1.000000 1.000000"]


def test_auto_control_runs_on_its_own_thread_at_the_configured_rate(live_backend, fake_port, stub_proposer):
    live_backend.set_auto_control(enabled=True, period_ms=100)
    time.sleep(1.5)
    live_backend.set_auto_control(enabled=False)
    sent = len(fake_port.commands("TARGETS"))

    # 15 at a perfect 100 ms. The old dashboard callback could never exceed 1 per second.
    assert 10 <= sent <= 16

    time.sleep(0.4)
    assert len(fake_port.commands("TARGETS")) == sent, "kept actuating after being disabled"


def test_auto_control_stops_steering_when_the_board_goes_quiet(backend_module, live_backend, fake_port, stub_proposer, monkeypatch):
    monkeypatch.setattr(backend_module, "AUTO_CONTROL_MAX_SAMPLE_AGE_SEC", 0.3)
    live_backend.set_auto_control(enabled=True, period_ms=100)
    assert wait_for(lambda: fake_port.commands("TARGETS"))

    fake_port.streaming.clear()
    time.sleep(0.6)
    sent = len(fake_port.commands("TARGETS"))
    time.sleep(0.5)

    assert len(fake_port.commands("TARGETS")) == sent
    assert live_backend.get_auto_control()["state"] == "waiting"
