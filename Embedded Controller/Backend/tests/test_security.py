import importlib.util
import json
import re
from pathlib import Path

import pytest

import python_Audit as audit
import python_dashboard_script as dashboard

BACKEND_DIR = Path(__file__).resolve().parents[1]
FIRMWARE_DIR = BACKEND_DIR.parent
MAIN_CPP = (FIRMWARE_DIR / "src" / "main.cpp").read_text(encoding="utf-8")
PLATFORMIO_INI = (FIRMWARE_DIR / "platformio.ini").read_text(encoding="utf-8")


def load_sbom_tool():
    spec = importlib.util.spec_from_file_location("generate_sbom", FIRMWARE_DIR / "tools" / "generate_sbom.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module



@pytest.mark.req("SEC-07")
def test_the_sbom_lists_every_pin_the_project_declares():
    tool = load_sbom_tool()
    sbom = tool.build_sbom()
    purls = {component["purl"] for component in sbom["components"]}

    assert sbom["bomFormat"] == "CycloneDX" and sbom["specVersion"] == "1.5"
    assert len(purls) == len(sbom["components"]), "every component appears once"

    for line in (BACKEND_DIR / "requirements.txt").read_text().splitlines() + (BACKEND_DIR / "requirements-dev.txt").read_text().splitlines():
        if "==" in line and not line.startswith("#"):
            name, version = line.split("==")
            assert f"pkg:pypi/{name.strip().lower()}@{version.strip()}" in purls

    assert "pkg:platformio/espressif32@6.12.0" in purls
    assert "pkg:platformio/adafruit/Adafruit%20ADS1X15@2.6.2" in purls
    assert all(" " not in purl for purl in purls), "package URLs are percent-encoded"


@pytest.mark.req("SEC-07")
def test_the_sbom_tool_writes_a_file(tmp_path):
    output = tmp_path / "sbom.json"
    load_sbom_tool().main(["generate_sbom.py", str(output)])
    assert json.loads(output.read_text())["components"]



@pytest.fixture
def secured(monkeypatch, tmp_path):
    monkeypatch.setattr(dashboard, "DASHBOARD_PASSWORD", "op-secret")
    monkeypatch.setattr(dashboard, "DASHBOARD_OBSERVER_PASSWORD", "obs-secret")
    log_path = tmp_path / "audit.jsonl"
    monkeypatch.setenv("TESTBED_AUDIT_LOG", str(log_path))
    return log_path


def post_callback(client, changed, inputs, auth):
    """Fires the safety controls callback the way the browser does, returning the raw response."""

    payload = {"output": "..safety-status.children...safety-status.style..",
               "outputs": [{"id": "safety-status", "property": "children"}, {"id": "safety-status", "property": "style"}],
               "changedPropIds": [changed], "inputs": inputs, "state": []}
    return client.post("/_dash-update-component", json=payload, auth=auth)


ARM_INPUTS = [{"id": "arm-confirm", "property": "submit_n_clicks", "value": 1},
              {"id": "disarm-button", "property": "n_clicks", "value": None},
              {"id": "clear-faults-button", "property": "n_clicks", "value": None},
              {"id": "update-interval", "property": "n_intervals", "value": 0}]


@pytest.mark.req("SEC-01")
def test_an_observer_can_watch_and_make_safe_but_never_make_live(shared_backend, secured):
    client = dashboard.server.test_client()
    observer = ("observer", "obs-secret")

    assert client.get("/", auth=observer).status_code == 200
    assert post_callback(client, "arm-confirm.submit_n_clicks", ARM_INPUTS, observer).status_code == 403
    assert post_callback(client, "update-interval.n_intervals", ARM_INPUTS, observer).status_code == 200

    disarm = [dict(item, value=1) if item["id"] == "disarm-button" else dict(item, value=None) if item["id"] == "arm-confirm" else item
              for item in ARM_INPUTS]
    assert post_callback(client, "disarm-button.n_clicks", disarm, observer).status_code == 200
    assert shared_backend.sent == ["DISARM"]


@pytest.mark.req("SEC-01", "SEC-03")
def test_an_operator_can_arm_and_every_action_is_audited(shared_backend, secured):
    client = dashboard.server.test_client()

    assert post_callback(client, "arm-confirm.submit_n_clicks", ARM_INPUTS, ("operator", "op-secret")).status_code == 200
    assert shared_backend.sent[0] == "ARM"

    post_callback(client, "arm-confirm.submit_n_clicks", ARM_INPUTS, ("observer", "obs-secret"))
    post_callback(client, "update-interval.n_intervals", ARM_INPUTS, ("operator", "op-secret"))   # ticks are not audited

    entries = [json.loads(line) for line in secured.read_text().splitlines()]
    dashboard_entries = [entry for entry in entries if entry["source"] == "dashboard"]

    assert [(e["user"], e["role"], e.get("result")) for e in dashboard_entries] == [("operator", "operator", None), ("observer", "observer", "refused")]
    assert dashboard_entries[0]["inputs"] == {"arm-confirm.submit_n_clicks": 1}
    assert {"source": "backend", "action": "ARM"}.items() <= next(e for e in entries if e["source"] == "backend").items()


@pytest.mark.req("SEC-01")
def test_wrong_or_empty_observer_credentials_are_refused(shared_backend, secured, monkeypatch):
    client = dashboard.server.test_client()
    assert client.get("/", auth=("observer", "wrong")).status_code == 401

    monkeypatch.setattr(dashboard, "DASHBOARD_OBSERVER_PASSWORD", "")
    assert client.get("/", auth=("observer", "")).status_code == 401


@pytest.mark.req("SEC-01")
def test_every_input_that_changes_the_rig_is_operator_only():
    # Any input of a callback that sends a command must be classed; only the safe ones are open.
    source = (BACKEND_DIR / "python_dashboard_script.py").read_text(encoding="utf-8")
    inputs = {f"{component}.{prop}" for component, prop in re.findall(r'Input\("([a-z0-9-]+)", "([a-z_]+)"\)', source)}
    open_to_observers = {"update-interval.n_intervals", "disarm-button.n_clicks", "sweep-stop-btn.n_clicks",
                         "plot-window-samples.value", "apply-plot-window.n_clicks", "buffer-samples.value"}

    assert inputs - open_to_observers == dashboard.OPERATOR_ONLY_INPUTS


@pytest.mark.req("SEC-02")
@pytest.mark.parametrize(
    "host, password, cert, key, plaintext, refused",
    [
        ("127.0.0.1", "", "", "", False, None),
        ("0.0.0.0", "", "", "", False, "without a password"),
        ("0.0.0.0", "pw", "", "", False, "without TLS"),
        ("0.0.0.0", "pw", "cert.pem", "", False, "without TLS"),
        ("0.0.0.0", "pw", "", "", True, None),
        ("0.0.0.0", "pw", "cert.pem", "key.pem", False, None),
    ],
)
def test_the_dashboard_only_serves_off_loopback_with_a_password_and_tls(host, password, cert, key, plaintext, refused):
    refusal = dashboard._launch_refusal(host, password, cert, key, allow_plaintext=plaintext)
    assert (refusal is None) if refused is None else refused in refusal


@pytest.mark.req("SEC-03")
def test_the_audit_trail_never_stops_the_rig(monkeypatch, tmp_path):
    monkeypatch.setenv("TESTBED_AUDIT_LOG", str(tmp_path / "missing-dir" / "audit.jsonl"))
    audit.record({"source": "test"})   # the directory does not exist: nothing raised



@pytest.mark.req("SEC-04")
def test_a_saved_dataset_carries_what_produced_it_and_its_hash(backend_module, tmp_path):
    import pandas as pd

    backend = backend_module.SerialBackend(port="FAKE0", status=True)
    backend.board_version = {"firmware": "abc1234", "protocol": "2"}
    backend.switch_timing = 5.0
    frame = pd.DataFrame({"voltage": [1.0, 1.1], "source": ["hardware", "hardware"]})
    csv_path = tmp_path / "sweep_dataset_20261006_120000.csv"
    frame.to_csv(csv_path, index=False)

    sidecar = backend._write_dataset_provenance(csv_path, frame, {"step_v": 0.5})
    details = json.loads(sidecar.read_text())

    assert sidecar.name == "sweep_dataset_20261006_120000.json"
    assert details["sha256"] == audit.file_sha256(csv_path)
    assert details["board"]["firmware"] == "abc1234" and details["rows"] == 2 and details["sources"] == {"hardware": 2}
    assert details["switch_period_us"] == 5.0 and details["sweep"] == {"step_v": 0.5}
    assert details["backend_revision"]


@pytest.mark.req("SEC-04")
def test_a_model_checkpoint_carries_its_training_provenance(tmp_path, monkeypatch):
    import python_RNN_Controller as rnn
    from sklearn.preprocessing import StandardScaler
    from helpers import sweep_shaped_frame

    monkeypatch.setattr(rnn, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(rnn, "MODEL_PATH", tmp_path / f"{rnn.MODEL_BASENAME}.pt")
    frame = sweep_shaped_frame()
    rnn.set_training_provenance(audit.provenance({"firmware": "abc1234"}, dataset_sha256=audit.frame_sha256(frame)))
    rnn.train_model(frame, number_of_epochs=1)
    rnn.save_nn_weights(rnn.model, rnn.scaler)
    saved = rnn.get_training_provenance()

    rnn.set_training_provenance({})
    assert rnn.load_previous_weights(rnn._RNN(rnn.INPUT_SIZE, rnn.HIDDEN_SIZE, rnn.OUTPUT_SIZE), StandardScaler())
    assert rnn.get_training_provenance() == saved and saved["board"]["firmware"] == "abc1234"


@pytest.mark.req("SEC-04")
def test_the_dataset_hash_ignores_the_index_but_not_the_values():
    import pandas as pd

    frame = pd.DataFrame({"voltage": [1.0, 2.0]})
    assert audit.frame_sha256(frame) == audit.frame_sha256(frame.set_index(pd.Index([7, 8])))
    assert audit.frame_sha256(frame) != audit.frame_sha256(pd.DataFrame({"voltage": [1.0, 2.5]}))



@pytest.mark.req("SEC-05", "OUT-03")
def test_the_deploy_build_has_no_radio_and_expects_the_verification_hardware():
    deploy = PLATFORMIO_INI[PLATFORMIO_INI.index("[env:esp32_deploy]"):]
    for flag in ("-DTESTBED_PRODUCTION=1", "-DTESTBED_GATE_LOOPBACK=1", "-DTESTBED_SETPOINT_READBACK=1", "-DLOG_LEVEL=0"):
        assert flag in deploy

    assert "constexpr bool WIRELESS_ENABLED = TESTBED_PRODUCTION == 0;" in MAIN_CPP
    setup, loop = MAIN_CPP[MAIN_CPP.index("\nvoid setup()"):MAIN_CPP.index("\nvoid loop()")], MAIN_CPP[MAIN_CPP.index("\nvoid loop()"):]
    assert re.search(r"if \(WIRELESS_ENABLED\) \{\s+ota_wifi_service\.begin\(", setup)
    assert re.search(r"if \(WIRELESS_ENABLED\) \{\s+ota_wifi_service\.loop\(", loop)
    assert "ota_wifi_service." not in re.sub(r"if \(WIRELESS_ENABLED\) \{\s+ota_wifi_service\.\w+\([^;]*\);", "", setup + loop)


@pytest.mark.req("SEC-06")
def test_an_updated_image_is_kept_only_if_its_self_test_passes():
    assert "bool verifyRollbackLater() {\n  return true;\n}" in MAIN_CPP

    setup = MAIN_CPP[MAIN_CPP.index("\nvoid setup()"):MAIN_CPP.index("\nvoid loop()")]
    assert setup.index("supervisor.begin();") < setup.index("confirm_or_roll_back_image();") < setup.index("ota_wifi_service.begin(")

    confirm = MAIN_CPP[MAIN_CPP.index("void confirm_or_roll_back_image()"):]
    confirm = confirm[:confirm.index("\n}\n")]
    assert "ESP_OTA_IMG_PENDING_VERIFY" in confirm
    assert confirm.index("supervisor.image_checks_passed()") < confirm.index("esp_ota_mark_app_valid_cancel_rollback();")
    assert "esp_ota_mark_app_invalid_rollback_and_reboot();" in confirm

    checks = MAIN_CPP[MAIN_CPP.index("bool image_checks_passed() const"):]
    checks = checks[:checks.index("\n  }\n")]
    assert "ClockConfig" in checks and "SwitchGenerator" in checks and "AdcLost" not in checks
