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


def test_a_switch_period_is_never_rounded_and_never_clamped_quietly(live_backend, fake_port, caplog):
    live_backend.set_switch_timing(7.5)
    assert fake_port.commands("SWITCH_PERIOD_US") == []
    assert "7.5 us is not a whole number" in caplog.text

    live_backend.set_switch_timing(3_000_000)
    assert fake_port.commands("SWITCH_PERIOD_US") == ["SWITCH_PERIOD_US 2000000"]
    assert "clamped to 2000000 us" in caplog.text

    live_backend.set_switch_timing(7.0)
    assert fake_port.commands("SWITCH_PERIOD_US")[-1] == "SWITCH_PERIOD_US 7"


def test_the_1_us_switch_floor_reaches_the_board(live_backend, fake_port, caplog):
    live_backend.set_switch_timing(1)
    live_backend.set_switch_timing(5)

    assert fake_port.commands("SWITCH_PERIOD_US") == ["SWITCH_PERIOD_US 1", "SWITCH_PERIOD_US 5"]
    assert "clamped" not in caplog.text


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


def test_a_sweep_trains_on_everything_it_recorded_not_just_the_buffer(backend_module, armed_backend, monkeypatch):
    live_backend = armed_backend
    trained_on = []
    monkeypatch.setattr(backend_module, "train_model", lambda frame, number_of_epochs=1: trained_on.append(frame) or {"loss": 0.0, "r2": 0.0})

    live_backend.set_buffer_samples(backend_module.DATA_BUFFER_MIN_SAMPLES)
    run_sweep(live_backend)

    assert live_backend.get_sweep_status()["state"] == "completed"
    assert len(trained_on[0]) > backend_module.DATA_BUFFER_MIN_SAMPLES
    assert set(trained_on[0]["source"]) == {"hardware"}


def test_an_aborted_sweep_closes_its_capture_and_skips_training(backend_module, armed_backend, monkeypatch):
    live_backend = armed_backend
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


def test_auto_control_acts_only_when_enabled_armed_fresh_and_not_sweeping(backend_module, stub_proposer, fake_port, validated_model):
    backend = backend_module.SerialBackend(port="FAKE0", status=False)
    backend.connect()

    assert backend._auto_control_step() is False
    assert backend.get_auto_control()["state"] == "off"

    backend.set_auto_control(enabled=True, change_penalty=0.3)
    assert backend._auto_control_step() is False
    assert "not armed" in backend.get_auto_control()["message"]

    backend._handle_board_line(time.time(), "ACK ARM")
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
    # The stub asks for 1 V from 0 V; the guard moves each channel one 0.25 V step.
    assert fake_port.commands("TARGETS") == ["TARGETS 0.250000 0.250000 0.250000 0.250000 0.250000"]


def test_auto_control_runs_on_its_own_thread_at_the_configured_rate(armed_backend, fake_port, stub_proposer, validated_model):
    live_backend = armed_backend
    live_backend.set_auto_control(enabled=True, period_ms=100)
    time.sleep(1.5)
    live_backend.set_auto_control(enabled=False)
    sent = len(fake_port.commands("TARGETS"))

    # 15 at a perfect 100 ms. The old dashboard callback could never exceed 1 per second.
    assert 10 <= sent <= 16

    time.sleep(0.4)
    assert len(fake_port.commands("TARGETS")) == sent, "kept actuating after being disabled"


def test_auto_control_stops_steering_when_the_board_goes_quiet(backend_module, armed_backend, fake_port, stub_proposer, monkeypatch, validated_model):
    live_backend = armed_backend
    monkeypatch.setattr(backend_module, "AUTO_CONTROL_MAX_SAMPLE_AGE_SEC", 0.3)
    live_backend.set_auto_control(enabled=True, period_ms=100)
    assert wait_for(lambda: fake_port.commands("TARGETS"))

    fake_port.streaming.clear()
    time.sleep(0.6)
    sent = len(fake_port.commands("TARGETS"))
    time.sleep(0.5)

    assert len(fake_port.commands("TARGETS")) == sent
    assert live_backend.get_auto_control()["state"] == "waiting"


# ----------------------------------------------------------------------------
#                               Safety state
# ----------------------------------------------------------------------------


def test_arming_follows_what_the_board_says(live_backend, fake_port):
    assert live_backend.board_mode in ("UNKNOWN", "SAFE")

    live_backend.arm()
    assert wait_for(live_backend.is_armed)

    live_backend.disarm()
    assert wait_for(lambda: live_backend.board_mode == "SAFE")


def test_a_refused_arm_is_recorded_and_leaves_the_board_unarmed(live_backend, fake_port):
    fake_port.mode = "FAULT"
    live_backend.arm()

    assert wait_for(lambda: "Cannot arm" in live_backend.get_board_safety()["last_arm_refusal"])
    assert not live_backend.is_armed()

    live_backend.clear_faults()
    assert wait_for(lambda: live_backend.board_mode == "SAFE")


def test_fault_reports_and_the_failsafe_update_the_mode(live_backend, fake_port):
    fake_port.lines.put("FAULTS mode=FAULT active=none latched=UNEXPECTED_RESET history=UNEXPECTED_RESET "
                        "counts=UNEXPECTED_RESET:1 boots=7 unexpected_resets=1 last_reset=TASK_WDT")
    assert wait_for(lambda: live_backend.board_mode == "FAULT")
    assert live_backend.get_board_safety()["faults"]["last_reset"] == "TASK_WDT"

    fake_port.lines.put("FAULT HOST_TIMEOUT WARNING mode=FAULT")
    assert wait_for(lambda: (live_backend.get_board_safety()["last_fault"] or {}).get("fault") == "HOST_TIMEOUT")

    live_backend.board_mode = "ARMED"
    fake_port.lines.put("FAILSAFE outputs zeroed, no command received from host")
    assert wait_for(lambda: live_backend.board_mode == "SAFE")


def test_the_keepalive_also_keeps_the_fault_report_current(fast_keepalive, live_backend, fake_port):
    assert wait_for(lambda: len(fake_port.commands("FAULTS")) >= 2)
    assert live_backend.get_board_safety()["faults"].get("boots") == "1"


def test_a_sweep_needs_the_board_armed(backend_module, live_backend, monkeypatch):
    monkeypatch.setattr(backend_module, "train_model", lambda *args, **kwargs: {"loss": 0.0, "r2": 0.0})

    assert live_backend.start_training_sweep(voltage_step_size=0.5, step_linger_time=0.01, epochs=1) is False
    assert "not armed" in live_backend.get_sweep_status()["message"]


def test_stopping_the_backend_disarms_the_board(backend_module, fake_port):
    backend = backend_module.SerialBackend(port="FAKE9", status=False)
    backend.online_update_enabled = False
    backend.start()
    assert wait_for(lambda: fake_port.opened == ["FAKE9"])

    backend.stop()
    assert fake_port.commands("DISARM") == ["DISARM"]


def test_a_dropped_link_forgets_the_mode(armed_backend):
    armed_backend.disconnect()
    assert armed_backend.board_mode == "UNKNOWN"
    assert not armed_backend.is_armed()


def test_a_simulated_board_arms_and_disarms_like_a_real_one(backend_module):
    backend = backend_module.SerialBackend(port="FAKE0", status=True)
    assert backend.board_mode == "SAFE"

    backend.arm()
    assert backend.is_armed()

    backend.update_pin_values(1, 500)
    backend.disarm()
    assert backend.board_mode == "SAFE" and backend.pins[0] == 0


def test_a_disconnected_board_is_never_reported_armed(backend_module):
    backend = backend_module.SerialBackend(port="FAKE0", status=False)
    backend.arm()  # no port open, so this goes down the offline path

    assert not backend.is_armed()


# ----------------------------------------------------------------------------
#                               Built-in test
# ----------------------------------------------------------------------------


def test_the_keepalive_polls_the_board_health(fast_keepalive, live_backend, fake_port):
    assert wait_for(lambda: len(fake_port.commands("HEALTH")) >= 2)
    health = live_backend.get_board_health()["health"]
    assert health["loop_budget_us"] == "20000" and health["adc"] == "ok"


def test_a_self_test_result_is_recorded_and_a_failure_logged(live_backend, fake_port, caplog):
    live_backend.run_selftest()
    assert wait_for(lambda: (live_backend.get_board_health()["selftest"] or {}).get("result") == "PASS")

    fake_port.selftest = "FAIL"
    live_backend.run_selftest()
    assert wait_for(lambda: live_backend.get_board_health()["selftest"]["result"] == "FAIL")
    assert "self-test failed" in caplog.text


# ----------------------------------------------------------------------------
#                              Link integrity
# ----------------------------------------------------------------------------


def test_every_command_goes_out_with_a_crc(armed_backend, fake_port):
    armed_backend.set_pin_voltages([1.0, 0.5, 0.0, 3.3, 2.0])
    armed_backend.set_switch_timing(5)

    assert fake_port.raw_written and all(valid for _, valid in fake_port.raw_written)
    assert fake_port.commands("TARGETS") == ["TARGETS 1.000000 0.500000 0.000000 3.300000 2.000000"]


def test_the_backend_learns_the_board_version_on_connect(live_backend):
    assert wait_for(lambda: live_backend.protocol_ok is True)
    assert live_backend.get_link_health()["version"]["firmware"] == "fake"


def test_a_board_speaking_another_protocol_is_never_armed(backend_module, fake_port):
    fake_port.protocol = 1
    backend = backend_module.SerialBackend(port="FAKE0", status=False)
    backend.online_update_enabled = False
    backend.start()

    try:
        assert wait_for(lambda: backend.protocol_ok is False)
        backend.arm()
        time.sleep(0.2)

        assert fake_port.commands("ARM") == []
        assert not backend.is_armed()
        assert "protocol 1" in backend.get_board_safety()["last_arm_refusal"]
    finally:
        backend.stop()


def test_corrupted_and_unchecksummed_lines_are_dropped_and_counted(live_backend, fake_port):
    assert wait_for(lambda: live_backend.protocol_ok is True)

    fake_port.put_raw("MEASURED 9.99999 V seq=1 t_ms=1*0000")       # CRC does not match
    fake_port.put_raw("MEASURED 8.88888 V seq=2 t_ms=2")            # protocol line, no CRC
    fake_port.put_raw("WiFi associating...")                        # free text needs none

    assert wait_for(lambda: live_backend.link_stats["bad_checksums"] == 1 and live_backend.link_stats["unverified"] == 1)
    assert not (live_backend.get_data()["voltage"] > 5).any()


def test_missing_readings_and_board_restarts_are_counted(live_backend, fake_port):
    fake_port.streaming.clear()
    time.sleep(0.2)
    last = live_backend._last_measured_seq
    missing_before = live_backend.link_stats["measured_missing"]

    for sequence in (last + 1, last + 2, last + 6, last + 7, 1):     # three lost, then a restart
        fake_port.lines.put(f"MEASURED 1.00000 V seq={sequence} t_ms={sequence * 50}")

    assert wait_for(lambda: live_backend.link_stats["board_restarts"] == 1)
    assert live_backend.link_stats["measured_missing"] - missing_before == 3


def test_replies_rejections_and_silence_are_all_accounted_for(live_backend, fake_port):
    assert wait_for(lambda: live_backend.protocol_ok is True)
    before = dict(live_backend.link_stats)

    live_backend.request_faults()                         # answered
    fake_port.mode = "FAULT"
    live_backend.arm()                                    # answered with an ERROR
    assert wait_for(lambda: live_backend.link_stats["rejections"] == before["rejections"] + 1)
    assert live_backend.link_stats["replies"] >= before["replies"] + 1

    live_backend.send_command("PINS")                     # the fake board never answers PINS
    live_backend._expire_pending_replies(now=time.time() + 5)
    assert live_backend.link_stats["reply_timeouts"] >= before["reply_timeouts"] + 1
    assert live_backend.get_link_health()["outstanding_replies"] == 0


def test_unsolicited_errors_never_settle_a_command(backend_module):
    backend = backend_module.SerialBackend(port="FAKE0", status=False)
    backend._pending_replies.append((time.time(), "ARM", backend_module.COMMAND_REPLIES["ARM"]))

    backend._match_reply("ERROR: ADC conversion timed out!")
    backend._match_reply("MEASURED 1.0 V seq=3 t_ms=150")
    assert len(backend._pending_replies) == 1

    backend._match_reply("ACK ARM")
    assert len(backend._pending_replies) == 0 and backend.link_stats["replies"] == 1


def test_frame_and_unframe_round_trip(backend_module):
    framed = backend_module.frame_line("HEALTH")
    assert backend_module.unframe_line(framed) == ("HEALTH", "valid")
    assert backend_module.unframe_line(framed[:-1] + ("0" if framed[-1] != "0" else "1"))[1] == "invalid"
    assert backend_module.unframe_line("WiFi associating...") == ("WiFi associating...", "absent")


# ----------------------------------------------------------------------------
#                          Auto control safeguards
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("validation, reason", [
    ({}, "no held-out validation score"),
    ({"validation_r2": -6.0}, "below the 0.50 floor"),
])
def test_auto_control_is_refused_without_a_validated_model(backend_module, monkeypatch, validation, reason):
    monkeypatch.setattr(backend_module, "get_validation_metrics", lambda: validation)
    backend = backend_module.SerialBackend(port="FAKE0", status=True)

    state = backend.set_auto_control(enabled=True)

    assert state["enabled"] is False and state["state"] == "refused" and reason in state["message"]


@pytest.fixture
def stepping_backend(backend_module, fake_port, validated_model):
    backend = backend_module.SerialBackend(port="FAKE0", status=False)
    backend.connect()
    backend._handle_board_line(time.time(), "ACK ARM")
    backend.update_pin_values(1, backend.voltage_to_pulse(1.0))
    backend._append_measurement(time.time(), 1.0, "MEASURED 1.0 V")
    backend.set_auto_control(enabled=True)
    return backend


def test_a_wild_proposal_is_held_to_the_envelope_and_the_step_limit(backend_module, stepping_backend, fake_port, monkeypatch):
    monkeypatch.setattr(backend_module, "propose_control_vector", lambda frame, change_penalty=0.1: [9.0, -4.0, 1.0, 3.3, 0.0])

    assert stepping_backend._auto_control_step() is True
    sent = [float(v) for v in fake_port.commands("TARGETS")[-1].split()[1:]]

    # Pin 1 starts at 1.0 V and pins 2-5 at 0 V; nothing moves more than 0.25 V or leaves 0-3.3 V.
    assert sent == pytest.approx([1.25, 0.0, 0.25, 0.25, 0.0], abs=0.005)
    assert "limited" in stepping_backend.get_auto_control()["message"].lower()
    assert stepping_backend.get_auto_control()["guard"]["limited"] == 1


def test_a_malformed_proposal_sends_nothing(backend_module, stepping_backend, fake_port, monkeypatch):
    monkeypatch.setattr(backend_module, "propose_control_vector", lambda frame, change_penalty=0.1: [1.0, float("nan"), 1.0, 1.0, 1.0])
    before = len(fake_port.commands("TARGETS"))

    assert stepping_backend._auto_control_step() is False
    assert len(fake_port.commands("TARGETS")) == before
    assert stepping_backend.get_auto_control()["state"] == "holding"


def test_inputs_outside_the_training_data_hold_the_rig(backend_module, stepping_backend, fake_port, monkeypatch, stub_proposer):
    stats = {"columns": ["pin_1", "pin_2", "pin_3", "pin_4", "pin_5", "voltage"], "mean": [100.0] * 5 + [1.0], "scale": [10.0] * 5 + [0.1]}
    monkeypatch.setattr(backend_module, "get_training_feature_stats", lambda: stats)
    before = len(fake_port.commands("TARGETS"))

    assert stepping_backend._auto_control_step() is False
    assert len(fake_port.commands("TARGETS")) == before and stub_proposer == []
    assert "drift" in stepping_backend.get_auto_control()["message"].lower()
