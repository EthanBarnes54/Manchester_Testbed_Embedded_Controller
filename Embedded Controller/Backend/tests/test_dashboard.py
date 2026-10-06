"""The dashboard, driven through Dash's own callback endpoint so callback context is real."""

import json

import numpy as np
import pandas as pd
import pytest

import python_dashboard_script as dashboard


@pytest.fixture
def client(shared_backend):
    test_client = dashboard.server.test_client()
    assert test_client.get("/").status_code == 200
    return test_client


def fire(client, outputs, inputs, changed, state=()):
    """Posts one callback the way the browser does and returns its response by component id."""

    if len(outputs) == 1:
        component, prop = outputs[0]
        output, output_spec = f"{component}.{prop}", {"id": component, "property": prop}
    else:
        output = ".." + "...".join(f"{component}.{prop}" for component, prop in outputs) + ".."
        output_spec = [{"id": component, "property": prop} for component, prop in outputs]

    payload = {
        "output": output,
        "outputs": output_spec,
        "changedPropIds": [changed] if changed else [],  # a page load triggers nothing
        "inputs": [{"id": component, "property": prop, "value": value} for component, prop, value in inputs],
        "state": [{"id": component, "property": prop, "value": value} for component, prop, value in state],
    }

    reply = client.post("/_dash-update-component", json=payload)
    assert reply.status_code in (200, 204), reply.get_data(as_text=True)[:500]
    return json.loads(reply.get_data(as_text=True))["response"] if reply.status_code == 200 else {}


def layout_component(client, component_id):
    def search(node):
        if isinstance(node, dict):
            if node.get("props", {}).get("id") == component_id:
                return node["props"]
            children = node.values()
        elif isinstance(node, list):
            children = node
        else:
            return None

        for child in children:
            found = search(child)
            if found is not None:
                return found

        return None

    return search(json.loads(client.get("/_dash-layout").get_data(as_text=True)))


# ----------------------------------------------------------------------------
#                                 Access
# ----------------------------------------------------------------------------


def test_a_configured_password_gates_every_request(client, monkeypatch):
    monkeypatch.setattr(dashboard, "DASHBOARD_PASSWORD", "s3cret")

    assert client.get("/").status_code == 401
    assert client.get("/", auth=("operator", "wrong")).status_code == 401
    assert client.get("/", auth=(dashboard.DASHBOARD_USER, "s3cret")).status_code == 200


# ----------------------------------------------------------------------------
#                              Control panel
# ----------------------------------------------------------------------------

PIN_OUTPUTS = [("pins-ack", "children"), ("pins-ack", "style")]
SAFETY_OUTPUTS = [("safety-status", "children"), ("safety-status", "style")]


@pytest.fixture
def armed_shared_backend(shared_backend):
    """The shared backend with its board reporting ARMED, as after an operator's ARM."""

    shared_backend.board_mode = "ARMED"
    return shared_backend


def safety_inputs(arm=None, disarm=None, clear=None):
    return [("arm-confirm", "submit_n_clicks", arm), ("disarm-button", "n_clicks", disarm),
            ("clear-faults-button", "n_clicks", clear), ("update-interval", "n_intervals", 0)]
DEFAULT_PINS = [("pwm1", "value", "1.0"), ("pwm2", "value", "0"), ("pwm3", "value", "0"), ("pwm4", "value", "0"), ("pwm5", "value", "0")]


def test_the_switch_field_uses_the_shared_bounds_and_starts_empty(client):
    switch = layout_component(client, "switch-time-us")

    assert (switch["min"], switch["max"]) == (dashboard.SWITCH_PERIOD_MIN_US, dashboard.SWITCH_PERIOD_MAX_US)
    assert switch.get("value") is None


@pytest.mark.parametrize("switch_us, accepted", [(1, True), (5, True), (0, False), (2_000_000, True), (2_000_001, False)])
def test_the_switch_field_accepts_1_us_to_2_s(client, armed_shared_backend, monkeypatch, switch_us, accepted):
    shared_backend = armed_shared_backend
    monkeypatch.setattr(shared_backend, "switch_timing", None)
    reply = fire(client, PIN_OUTPUTS, DEFAULT_PINS + [("switch-time-us", "value", switch_us)], "switch-time-us.value")
    message = reply["pins-ack"]["children"]

    if accepted:
        assert message.endswith(f"switch_time={switch_us:.1f} us")
        assert shared_backend.switch_timing == switch_us
    else:
        assert message == "ERROR: Switch time out of range (1-2000000 us)!"
        assert shared_backend.switch_timing is None


def test_a_fractional_switch_time_is_refused_not_truncated(client, armed_shared_backend, monkeypatch):
    shared_backend = armed_shared_backend
    monkeypatch.setattr(shared_backend, "switch_timing", None)
    reply = fire(client, PIN_OUTPUTS, DEFAULT_PINS + [("switch-time-us", "value", 7.5)], "switch-time-us.value")

    assert reply["pins-ack"]["children"] == "ERROR: Switch time must be a whole number of us!"
    assert shared_backend.switch_timing is None
    assert "border" in dashboard._validate_switch_time_input(7.5)
    assert "border" not in dashboard._validate_switch_time_input(7)


def test_editing_a_voltage_sends_targets_and_leaves_the_switch_alone(client, armed_shared_backend):
    shared_backend = armed_shared_backend
    reply = fire(client, PIN_OUTPUTS, DEFAULT_PINS + [("switch-time-us", "value", None)], "pwm1.value")

    assert reply["pins-ack"]["children"].startswith("Targets updated")
    assert shared_backend.sent == ["TARGETS 1.000000 0.000000 0.000000 0.000000 0.000000"]


def test_pin_edits_are_refused_until_the_board_is_armed(client, shared_backend):
    reply = fire(client, PIN_OUTPUTS, DEFAULT_PINS + [("switch-time-us", "value", 5)], "pwm1.value")

    assert reply["pins-ack"]["children"] == "ERROR: Board is SAFE - press ARM before setting outputs!"
    assert shared_backend.sent == []


@pytest.mark.parametrize(
    "button, inputs, sent",
    [
        ("arm-confirm.submit_n_clicks", safety_inputs(arm=1), ["ARM"]),
        ("disarm-button.n_clicks", safety_inputs(disarm=1), ["DISARM"]),
        ("clear-faults-button.n_clicks", safety_inputs(clear=1), ["CLEAR FAULTS", "FAULTS"]),
        ("update-interval.n_intervals", safety_inputs(), []),
    ],
)
def test_the_safety_buttons_send_their_commands_and_nothing_else_does(client, shared_backend, button, inputs, sent):
    fire(client, SAFETY_OUTPUTS, inputs, button)
    assert shared_backend.sent == sent


def test_the_safety_readout_mirrors_the_board(client, shared_backend):
    shared_backend.board_mode = "FAULT"
    shared_backend.board_faults = {"latched": "UNEXPECTED_RESET"}
    shared_backend.last_arm_refusal = "ERROR: Cannot arm - a critical fault is latched, send CLEAR FAULTS!"

    try:
        reply = fire(client, SAFETY_OUTPUTS, safety_inputs(), "update-interval.n_intervals")
        text = reply["safety-status"]["children"]

        assert text.startswith("Board: FAULT") and "UNEXPECTED_RESET" in text and "Cannot arm" in text
        assert reply["safety-status"]["style"]["color"] == dashboard.SAFETY_MODE_COLOURS["FAULT"]
    finally:
        shared_backend.board_faults = {}
        shared_backend.last_arm_refusal = ""


def test_the_self_test_button_asks_the_board(client, shared_backend):
    fire(client, [("health-status", "children"), ("link-status", "children")],
         [("selftest-button", "n_clicks", 1), ("update-interval", "n_intervals", 0)], "selftest-button.n_clicks")
    assert shared_backend.sent == ["SELFTEST"]


def test_the_health_readout_summarises_the_board_report():
    summary = dashboard._health_summary({
        "health": {"loop_max_us": "1200", "loop_peak_us": "4300", "loop_budget_us": "20000", "heap_free": "204800",
                   "adc": "lost", "adc_timeouts": "7"},
        "selftest": {"result": "FAIL"},
    })

    assert summary == "Health: Loop 1.2 ms, peak 4.3 (budget 20 ms) | Heap 200 kB | ADC lost (7 timeouts) | Self-test FAIL"
    assert dashboard._health_summary({"health": {}, "selftest": None}) == "Health: --"
    assert dashboard._health_summary({"health": {"gate": "FAIL", "readback": "off"}, "selftest": None}) ==         "Health: Output checks: gate FAIL, readback off"


def test_the_link_readout_names_the_firmware_and_what_went_wrong():
    link = {"version": {"firmware": "1fa3f43"}, "protocol_ok": True, "bad_checksums": 2, "measured_missing": 5, "reply_timeouts": 1}
    assert dashboard._link_summary(link) == \
        "Link: firmware 1fa3f43, protocol ok | 2 corrupted lines | 5 missing readings | 1 unanswered commands"

    assert dashboard._link_summary({"protocol_ok": None}).startswith("Link: waiting for VERSION |")
    assert "PROTOCOL MISMATCH" in dashboard._link_summary({"version": {"firmware": "old"}, "protocol_ok": False})


def test_momentum_edits_reach_the_model(client):
    import python_RNN_Controller as rnn

    settings = [("online-window-seconds", "value", 30), ("online-learning-rate", "value", 1e-3),
                ("online-momentum", "value", 0.5), ("optimiser-type", "value", "adam")]
    fire(client, [("online-update-config-status", "children")], settings, "online-momentum.value")

    assert rnn.get_momentum() == pytest.approx(0.5)
    rnn.set_momentum(rnn.MOMENTUM)


def test_editing_one_setting_does_not_rebuild_the_optimiser(client, shared_backend):
    import python_RNN_Controller as rnn

    before = rnn.optimiser
    settings = [("online-window-seconds", "value", 45), ("online-learning-rate", "value", 1e-3),
                ("online-momentum", "value", 0.9), ("optimiser-type", "value", "adam")]
    fire(client, [("online-update-config-status", "children")], settings, "online-window-seconds.value")

    assert shared_backend.get_window_update_time() == 45
    assert rnn.optimiser is before


def test_the_sweep_button_reaches_the_sweep_worker(client, armed_shared_backend, monkeypatch):
    shared_backend = armed_shared_backend
    received = []
    monkeypatch.setattr(shared_backend, "_RNN_training_sweeps", lambda *args: received.append(args))

    fire(
        client,
        [("sweep-status", "children"), ("sweep-btn", "disabled"), ("sweep-progress-inner", "style")],
        [("update-interval", "n_intervals", 0), ("sweep-btn", "n_clicks", 1), ("sweep-stop-btn", "n_clicks", 0)],
        "sweep-btn.n_clicks",
        state=[("sweep-min", "value", 0.5), ("sweep-max", "value", 2.5), ("sweep-step", "value", 0.25),
               ("sweep-dwell", "value", 0.01), ("sweep-epochs", "value", 3), ("sweep-baselines", "value", 2),
               ("sweep-factorials", "value", 2), ("sweep-random-samples", "value", 4)],
    )
    shared_backend.sweep_thread.join(timeout=5)

    assert received == [(0.5, 2.5, 0.25, 0.01, 3, 2, 2, 4)]


def test_opening_a_tab_does_not_switch_dataset_saving_off(client, shared_backend):
    outputs = [("save-dataset-button", "children"), ("save-dataset-button", "style")]

    def press(n_clicks, changed):
        inputs = [("save-dataset-button", "n_clicks", n_clicks), ("update-interval", "n_intervals", 0)]
        return fire(client, outputs, inputs, changed)["save-dataset-button"]["children"]

    assert press(1, "save-dataset-button.n_clicks") == "Save After Sweep: ON"
    assert press(0, None) == "Save After Sweep: ON"  # the initial call a newly opened tab makes
    assert shared_backend.save_dataset_enabled

    assert press(2, "save-dataset-button.n_clicks") == "Save After Sweep: OFF"
    assert not shared_backend.save_dataset_enabled


def test_a_failed_shap_run_tells_the_operator_why(client, monkeypatch):
    def fail(**kwargs):
        raise RuntimeError("Scaler not fitted. Train the model first...")

    monkeypatch.setattr(dashboard, "compute_feature_importance", fail)
    outputs = [("compute-shap-button", "children"), ("compute-shap-button", "style"), ("shap-status", "children"), ("ml-shap-bar", "figure")]
    reply = fire(client, outputs, [("compute-shap-button", "n_clicks", 1)], "compute-shap-button.n_clicks", state=[("shap-permutations", "value", 5)])

    assert "Scaler not fitted" in reply["shap-status"]["children"]


# ----------------------------------------------------------------------------
#                               Auto control
# ----------------------------------------------------------------------------

TOGGLE_OUTPUTS = [("auto-mode-button", "children"), ("auto-mode-button", "style"), ("auto-mode-status", "children")]


def toggle(client, n_clicks, changed):
    reply = fire(client, TOGGLE_OUTPUTS, [("auto-mode-button", "n_clicks", n_clicks), ("update-interval", "n_intervals", 1)], changed)
    return reply["auto-mode-button"]["children"]


def test_every_tab_mirrors_the_backend_auto_control_state(client, shared_backend, validated_model):
    assert toggle(client, 1, "auto-mode-button.n_clicks") == "Auto Control: ON"
    assert shared_backend.get_auto_control()["enabled"]

    # A second tab that has never been clicked still shows the truth on its next tick.
    assert toggle(client, 0, "update-interval.n_intervals") == "Auto Control: ON"

    assert toggle(client, 1, "auto-mode-button.n_clicks") == "Auto Control: OFF"
    assert not shared_backend.get_auto_control()["enabled"]


def test_auto_control_settings_reach_the_backend_and_new_tabs(client, shared_backend):
    fire(client, [("auto-rate-ms", "style")], [("auto-rate-ms", "value", 250), ("auto-change-penalty", "value", 0.1)], "auto-rate-ms.value")
    fire(client, [("auto-rate-ms", "style")], [("auto-rate-ms", "value", 250), ("auto-change-penalty", "value", 0.4)], "auto-change-penalty.value")

    assert shared_backend.get_auto_control()["period_ms"] == 250
    assert shared_backend.get_auto_control()["change_penalty"] == pytest.approx(0.4)
    assert layout_component(client, "auto-rate-ms")["value"] == 250
    assert layout_component(client, "auto-change-penalty")["value"] == pytest.approx(0.4)


def test_the_plot_callback_never_actuates(client, shared_backend, validated_model):
    shared_backend.set_auto_control(enabled=True, period_ms=100)

    for tick in range(3):
        fire(
            client,
            [("live-graph", "figure"), ("latest-voltage", "children"), ("num-points", "children")],
            [("update-interval", "n_intervals", tick)],
            "update-interval.n_intervals",
            state=[("plot-window-config", "data", {"mode": "all"})],
        )

    assert shared_backend.sent == []


# ----------------------------------------------------------------------------
#                               Plot history
# ----------------------------------------------------------------------------


def test_plot_history_is_bounded_and_contiguous(shared_backend, monkeypatch):
    monkeypatch.setattr(dashboard, "PLOT_HISTORY", dashboard.deque(maxlen=dashboard.PLOT_HISTORY_MIN_SAMPLES))
    newest = [0]

    def rolling_buffer():
        newest[0] += 1000
        index = np.arange(newest[0] - 1000, newest[0])
        return pd.DataFrame({"timestamp": index * 0.05, "voltage": np.sin(index / 100.0)})

    monkeypatch.setattr(shared_backend, "get_data", rolling_buffer)

    for tick in range(30):  # 30,000 samples, 25 minutes at 20 Hz
        dashboard.update_graph(tick, {"mode": "all"})

    timestamps = [point[0] for point in dashboard.PLOT_HISTORY]

    assert len(timestamps) == dashboard.PLOT_HISTORY_MIN_SAMPLES
    assert timestamps[-1] == pytest.approx((newest[0] - 1) * 0.05)
    assert np.allclose(np.diff(timestamps), 0.05)

    shared_backend.set_buffer_samples(50000)
    dashboard.update_graph(0, {"mode": "all"})

    # Growing the cap keeps every point already held, plus the 1,000 that call read.
    assert dashboard.PLOT_HISTORY.maxlen == 50000
    assert len(dashboard.PLOT_HISTORY) == dashboard.PLOT_HISTORY_MIN_SAMPLES + 1000
    assert np.allclose(np.diff([point[0] for point in dashboard.PLOT_HISTORY]), 0.05)


def test_the_auto_button_explains_a_refusal(client, shared_backend, backend_module, monkeypatch):
    monkeypatch.setattr(backend_module, "get_validation_metrics", lambda: {})
    reply = fire(client, [("auto-mode-button", "children"), ("auto-mode-button", "style"), ("auto-mode-status", "children")],
                 [("auto-mode-button", "n_clicks", 1), ("update-interval", "n_intervals", 0)], "auto-mode-button.n_clicks")

    assert reply["auto-mode-button"]["children"] == "Auto Control: OFF"
    assert "no held-out validation score" in reply["auto-mode-status"]["children"]


def test_a_manual_edit_takes_over_from_auto_control(client, armed_shared_backend, validated_model):
    armed_shared_backend.set_auto_control(enabled=True)
    reply = fire(client, PIN_OUTPUTS, DEFAULT_PINS + [("switch-time-us", "value", None)], "pwm1.value")

    assert not armed_shared_backend.get_auto_control()["enabled"]
    assert reply["pins-ack"]["children"].endswith("(auto control switched off by this manual edit)")
