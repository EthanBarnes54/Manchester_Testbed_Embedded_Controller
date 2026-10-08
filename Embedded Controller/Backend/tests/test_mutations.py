"""Mutation tests: each case breaks one safety check on purpose and expects the suite to fail.

A test that still passes with the check it guards removed proves nothing, so every case
here is a small, realistic slip (a condition dropped, an order swapped, a constant moved)
in the firmware, a header or the backend, paired with the tests that must catch it.

Firmware and header cases rebuild the affected harness from the mutated source in memory.
Backend cases copy the project to a scratch directory, apply the change there and run the
named tests in a fresh interpreter. A case whose anchor text no longer appears exactly once
fails, so the list cannot rot silently as the code changes.

Slow (a few minutes), so excluded from the default run: python -m pytest -m mutation
"""

import inspect
import itertools
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import test_adc_sampling
import test_firmware_contract
import test_line_protocol
import test_output_verifier
import test_security
import test_switch_line
import test_switch_timing
import test_health_stats
import test_output_checks
import test_safety_state
from host_build import FIRMWARE_DIR, INCLUDE_DIR, compile_harness, main_cpp, needs_compiler, run

pytestmark = [pytest.mark.mutation, pytest.mark.req("QA-04")]


def mutate(text, old, new):
    assert text.count(old) == 1, f"the mutation's anchor appears {text.count(old)} times; update the case: {old[:70]!r}"
    return text.replace(old, new)


def harness_fails(source, work, name, **options):
    """Whether a self-checking harness built from mutated source fails. A mutant that does
    not compile is a broken case, not a caught one."""

    built, binary = compile_harness(source, work, name, warnings_as_errors=False, **options)
    assert built.returncode == 0, f"the mutant does not compile, so it proves nothing:\n{built.stderr[-1500:]}"
    return run(binary).returncode != 0


def expand(function):
    """Every argument set pytest would call a test function with, from its parametrize marks."""

    sets = [{}]

    for mark in getattr(function, "pytestmark", []):
        if mark.name != "parametrize":
            continue

        names = [name.strip() for name in mark.args[0].split(",")] if isinstance(mark.args[0], str) else list(mark.args[0])
        values = [value.values if type(value).__name__ == "ParameterSet" else value for value in mark.args[1]]
        values = [value if len(names) > 1 else (value,) for value in values]
        sets = [{**existing, **dict(zip(names, value))} for existing, value in itertools.product(sets, values)]

    return sets


def failing_tests(module, fixtures):
    """Runs every test function in a module that needs only the given fixtures, and returns
    the names of those that fail."""

    failed = []

    for name, function in inspect.getmembers(module, inspect.isfunction):
        if not name.startswith("test_") or function.__module__ != module.__name__:
            continue

        needs = set(inspect.signature(function).parameters)

        for arguments in expand(function):
            missing = needs - set(arguments) - set(fixtures)

            if missing:
                break

            try:
                function(**arguments, **{key: fixtures[key] for key in needs - set(arguments)})
            except Exception:
                failed.append(name)
                break

    return failed


# ----------------------------------------------------------------------------
#                    Firmware: main.cpp, through the harnesses
# ----------------------------------------------------------------------------

def switch_line_fails(source, work):
    return harness_fails(test_switch_line.harness_source(source), work, "switch_line")


def adc_fails(source, work):
    return any(harness_fails(test_adc_sampling.harness_source(source), work, f"adc_{readback}",
                             defines={"TESTBED_SETPOINT_READBACK": readback}) for readback in (0, 1))


def output_verifier_fails(source, work):
    return harness_fails(test_output_verifier.harness_source(source), work, "output_verifier",
                         defines=test_output_verifier.fitted_defines(1))


def contract_fails(source, work, monkeypatch):
    """The text checks on main.cpp, in the contract and security tests. The host harnesses
    in those modules are left to their own cases."""

    failed = []

    for module in (test_firmware_contract, test_security):
        monkeypatch.setattr(module, "MAIN_CPP", source)
        failed += [name for name in failing_tests(module, {}) if not name.endswith("_on_the_host")]

    return bool(failed)


MAIN_CPP_CASES = [
    # The switch line
    ("hold() routes the pin before setting its level", switch_line_fails,
     "    write_pin(high);\n    route_pin(SIG_GPIO_OUT_IDX);", "    route_pin(SIG_GPIO_OUT_IDX);\n    write_pin(high);"),
    ("the timer ISR ignores the mode", switch_line_fails,
     "    if (mode_ == Mode::Interrupt) {\n      interrupt_high_", "    if (true) {\n      interrupt_high_"),
    ("no priming cycle before the handover", switch_line_fails, "|| !run_one_cycle_off_pin(period_us)", ""),
    ("the interrupt count is not restarted", switch_line_fails, "    timerWrite(timer_, 0);\n", ""),
    ("the LEDC readback is ignored", switch_line_fails,
     "return conf.clock_divider == waveform.divider_register() &&", "return true ||"),
    ("the pin is routed after the timer is released", switch_line_fails,
     "    route_pin(SWITCH_LEDC_SIGNAL);\n    conf.val = held & ~LEDC_HSTIMER0_PAUSE;",
     "    conf.val = held & ~LEDC_HSTIMER0_PAUSE;\n    route_pin(SWITCH_LEDC_SIGNAL);"),
    ("releasing the timer clobbers its divider", switch_line_fails,
     "    conf.val = held & ~LEDC_HSTIMER0_PAUSE;", "    conf.val = 0;"),
    ("the counter is not reset before the handover", switch_line_fails, "    conf.val = held | LEDC_HSTIMER0_RST;\n", ""),
    ("ARMED is not cleared at boot", switch_line_fails,
     "    GPIO.out_w1tc = SWITCH_PIN_MASK | SWITCH_ARMED_PIN_MASK;", "    GPIO.out_w1tc = SWITCH_PIN_MASK;"),
    # The ADC
    ("READ starts a conversion directly", adc_fails,
     "    reading_requested_ = true;\n  }", "    reading_requested_ = true;\n    adc_.startADCReading(MUX_BY_CHANNEL[0], false);\n  }"),
    ("a conversion never times out", adc_fails, "    if (elapsed_ms >= ADC_CONVERSION_TIMEOUT_MS) {", "    if (false) {"),
    ("the ADC is polled from the start of a conversion", adc_fails, "    if (elapsed_ms < ADC_FIRST_POLL_MS) {", "    if (false) {"),
    ("a lost ADC never recovers", adc_fails, "      lost_ = false;\n      ++conversions_;", "      ++conversions_;"),
    ("ADC timeouts are not counted", adc_fails, "      lost_ = true;\n      ++timeouts_;", "      lost_ = true;"),
    ("a readback starts outside its window", adc_fails,
     "      } else if (readback_due_ && now_ms - diode_started_ms_ <= READBACK_START_WINDOW_MS) {", "      } else if (readback_due_) {"),
    ("readback inputs never rotate", adc_fails,
     "        next_readback_input_ = static_cast<uint8_t>(next_readback_input_ % READBACK_INPUTS + 1);\n", ""),
    # Output verification
    ("a window spanning a change is judged", output_verifier_fails,
     "    const bool window_valid = line.generation == window_generation_ && armed == window_armed_;",
     "    const bool window_valid = true;"),
    ("edges through a closed gate are ignored", output_verifier_fails,
     "      stray_edges_ = last_window_edges_ > 0;", "      stray_edges_ = false;"),
    ("the gate level is not debounced", output_verifier_fails,
     "        gate_level_check_(GATE_MISMATCH_PASSES),", "        gate_level_check_(1),"),
    ("one readback is judged on every pass", output_verifier_fails,
     "      if (read_ms == 0 || read_ms == readback_checked_ms_[index]) {", "      if (read_ms == 0) {"),
    ("a failed pulse counter is ignored", output_verifier_fails,
     "    return GATE_LOOPBACK_FITTED && (!counter_ready_ || gate_level_check_.tripped() || stray_edges_);",
     "    return GATE_LOOPBACK_FITTED && (gate_level_check_.tripped() || stray_edges_);"),
    # Safety, built-in test and the link, through the contract checks
    ("PIN drives an output while disarmed", contract_fails,
     "        if (!supervisor_.armed() && value > 0) {\n          send_line(\"ERROR: Not armed!\");\n          return;\n        }\n\n", ""),
    ("ARM ignores a latched fault", contract_fails,
     "    const char* refusal = state_.arm_refusal();", "    const char* refusal = nullptr;"),
    ("the failsafe closes the gate last", contract_fails,
     "  void engage_safe_state() {\n    set_switch_armed(false);\n\n", "  void engage_safe_state() {\n"),
    ("no loop watchdog", contract_fails,
     "  enableLoopWDT();\n\n  LOG_INFO(\"Setup complete", "  LOG_INFO(\"Setup complete"),
    ("an update runs while armed", contract_fails,
     "ota_wifi_service.loop(!supervisor.armed());", "ota_wifi_service.loop(true);"),
    ("the heartbeat sits on a strapping pin", contract_fails,
     "constexpr int HEARTBEAT_PIN = 18;", "constexpr int HEARTBEAT_PIN = 12;"),
    ("ARMED sits on a strapping pin", contract_fails,
     "constexpr int SWITCH_ARMED_PIN = 23;", "constexpr int SWITCH_ARMED_PIN = 5;"),
    ("the self-test skips the clocks", contract_fails,
     "    const bool clocks_ok = clocks_as_designed();", "    const bool clocks_ok = true;"),
    ("loop passes are not timed", contract_fails, "    loop_timing_.record(pass_us);\n", ""),
    ("serial overflows are not reported", contract_fails, "          supervisor_.note_serial_overflow();\n", ""),
    ("a bad CRC is accepted", contract_fails,
     "if (line_protocol::verify(command.c_str(), command.length(), &body_length) == line_protocol::Check::Invalid) {",
     "if (false) {"),
    ("readings are not numbered", contract_fails, '" V seq=" + sequence_ + " t_ms=" + millis()', '" V"'),
    ("replies hold up the loop", contract_fails, "  Serial.setTxBufferSize(SERIAL_TX_BUFFER_BYTES);\n", ""),
    ("a reply bypasses the CRC", contract_fails, '    send_line("ACK DISARM");', '    Serial.println("ACK DISARM");'),
    ("an updated image is kept without its checks", contract_fails,
     "  if (supervisor.image_checks_passed()) {", "  if (true) {"),
    ("the deploy build keeps its radio", contract_fails,
     "  if (WIRELESS_ENABLED) {\n    ota_wifi_service.loop(", "  {\n    ota_wifi_service.loop("),
]


@needs_compiler
@pytest.mark.parametrize("name, fails, old, new", MAIN_CPP_CASES, ids=[case[0] for case in MAIN_CPP_CASES])
def test_a_broken_firmware_check_is_caught(tmp_path, monkeypatch, name, fails, old, new):
    mutant = mutate(main_cpp(), old, new)
    arguments = (mutant, tmp_path, monkeypatch) if fails is contract_fails else (mutant, tmp_path)
    assert fails(*arguments), f"nothing noticed: {name}"


@needs_compiler
def test_the_unmutated_firmware_passes_every_check_used_above(tmp_path, monkeypatch):
    """Without this, a checker that always fails would make every case look caught."""

    source = main_cpp()
    assert not switch_line_fails(source, tmp_path)
    assert not adc_fails(source, tmp_path)
    assert not output_verifier_fails(source, tmp_path)
    assert not contract_fails(source, tmp_path, monkeypatch)


# ----------------------------------------------------------------------------
#                         Firmware: the shared headers
# ----------------------------------------------------------------------------

def self_checking(module, name):
    def fails(include_dir, work):
        return harness_fails(module.HARNESS, work, name, include_dirs=(include_dir,))
    return fails


def line_protocol_fails(include_dir, work):
    import python_Backend

    harness = test_line_protocol.make_harness(work, include_dirs=(include_dir,), warnings_as_errors=False)
    return bool(failing_tests(test_line_protocol, {"harness": harness, "backend_module": python_Backend}))


def switch_timing_fails(include_dir, work):
    harness = test_switch_timing.make_harness(work, include_dirs=(include_dir,), warnings_as_errors=False)
    return bool(failing_tests(test_switch_timing, {"harness": harness}))


HEADER_CASES = [
    ("safety_state.h", "a critical fault never blocks arming", self_checking(test_safety_state, "safety_state"),
     "if ((faults & (1UL << index)) && severity_of", "if (false && severity_of"),
    ("safety_state.h", "a critical fault leaves the board armed", self_checking(test_safety_state, "safety_state"),
     "    mode_ = Mode::Fault;\n    return was_armed;", "    return was_armed;"),
    ("safety_state.h", "clearing drops active latches too", self_checking(test_safety_state, "safety_state"),
     "latched_ &= active_;", "latched_ = 0;"),
    ("safety_state.h", "a fault is counted on every raise", self_checking(test_safety_state, "safety_state"),
     "if ((active_ & bit) == 0 && counts_[index] < 0xFFFF)", "if (counts_[index] < 0xFFFF)"),
    ("safety_state.h", "DISARM leaves FAULT", self_checking(test_safety_state, "safety_state"),
     "if (mode_ != Mode::Armed) {\n      return false;", "if (mode_ == Mode::Safe) {\n      return false;"),
    ("safety_state.h", "a clock fault is only a warning", self_checking(test_safety_state, "safety_state"),
     "    case Fault::ClockConfig:\n    case Fault::SwitchGenerator:\n", "    case Fault::SwitchGenerator:\n"),
    ("safety_state.h", "a gate mismatch is only a warning", self_checking(test_safety_state, "safety_state"),
     "    case Fault::GateMismatch:\n    case Fault::SwitchFrequency:\n", "    case Fault::SwitchFrequency:\n"),
    ("health_stats.h", "a pass at the budget is an overrun", self_checking(test_health_stats, "health"),
     "pass_us > budget_us_ && overruns_", "pass_us >= budget_us_ && overruns_"),
    ("health_stats.h", "the reporting window never resets", self_checking(test_health_stats, "health"),
     "    window_max_us_ = 0;\n    return worst;", "    return worst;"),
    ("output_checks.h", "an edge per period, not per cycle", self_checking(test_output_checks, "output_checks"),
     "elapsed_us / (2ULL * period_us)", "elapsed_us / period_us"),
    ("output_checks.h", "a 5% edge count tolerance", self_checking(test_output_checks, "output_checks"),
     "expected / 200 > 2 ? expected / 200 : 2", "expected / 20 > 2 ? expected / 20 : 2"),
    ("output_checks.h", "a disarmed gate follows the line", self_checking(test_output_checks, "output_checks"),
     "  if (!armed) {\n    return GateExpectation::Low;\n  }\n", ""),
    ("line_protocol.h", "the CRC starts from zero", line_protocol_fails, "  uint16_t crc = 0xFFFF;", "  uint16_t crc = 0x0000;"),
    ("line_protocol.h", "lowercase hex is not accepted", line_protocol_fails,
     "    } else if (digit >= 'a' && digit <= 'f') {\n      value = static_cast<uint8_t>(digit - 'a' + 10);\n", ""),
    ("switch_timing.h", "a fractional period is rounded", switch_timing_fails,
     "  if (cycle_ticks_scaled % 1000000ULL != 0) {\n    return rejected;\n  }\n", ""),
    ("switch_timing.h", "an oversized divider is accepted", switch_timing_fails,
     "if (bits == 0 || divider > LEDC_MAX_DIVIDER) {", "if (bits == 0) {"),
]


@needs_compiler
@pytest.mark.parametrize("header, name, fails, old, new", HEADER_CASES, ids=[f"{case[0]}: {case[1]}" for case in HEADER_CASES])
def test_a_broken_header_check_is_caught(tmp_path, header, name, fails, old, new):
    include_dir = tmp_path / "include"
    include_dir.mkdir()
    (include_dir / header).write_text(mutate((INCLUDE_DIR / header).read_text(encoding="utf-8"), old, new), encoding="utf-8")
    assert fails(include_dir, tmp_path), f"nothing noticed: {name}"


@needs_compiler
def test_the_unmutated_headers_pass_every_check_used_above(tmp_path):
    for fails in {case[2] for case in HEADER_CASES}:
        assert not fails(INCLUDE_DIR, tmp_path)


# ----------------------------------------------------------------------------
#               Backend, dashboard and tools: in a scratch copy
# ----------------------------------------------------------------------------

BACKEND_CASES = [
    # The link
    ("Backend/python_Backend.py", "a corrupted line is accepted", '        if crc == "invalid":', "        if False:",
     "tests/test_backend.py -k corrupted"),
    ("Backend/python_Backend.py", "an unframed protocol line is accepted",
     '        if crc == "absent" and self.protocol_ok and text.startswith(PROTOCOL_PREFIXES):', "        if False:",
     "tests/test_backend_fuzz.py -k unframed"),
    ("Backend/python_Backend.py", "a non-finite reading is stored", "                if not math.isfinite(voltage):", "                if False:",
     "tests/test_backend_fuzz.py"),
    ("Backend/python_Backend.py", "a board on another protocol may be armed",
     "        if not self.force_offline and self.protocol_ok is not True:", "        if False:", "tests/test_backend.py -k protocol"),
    ("Backend/python_Backend.py", "an unsolicited error settles a command",
     '            elif message.startswith("ERROR") and not message.startswith(UNSOLICITED_ERRORS):',
     '            elif message.startswith("ERROR"):', "tests/test_backend.py -k unsolicited"),
    ("Backend/python_Backend.py", "missing readings are miscounted",
     '            self.link_stats["measured_missing"] += sequence - previous - 1',
     '            self.link_stats["measured_missing"] += sequence - previous', "tests/test_backend.py -k missing"),
    ("Backend/python_Backend.py", "commands go out without a CRC",
     '                self.serial.write((frame_line(command_string.strip()) + "\\n").encode("utf-8"))',
     '                self.serial.write((command_string.strip() + "\\n").encode("utf-8"))', "tests/test_backend.py -k crc"),
    # Arming
    ("Backend/python_Backend.py", "a sweep runs disarmed",
     '        # A sweep drives every output, so it needs the board armed first like any operator would.\n        if not self.is_armed():',
     '        # A sweep drives every output, so it needs the board armed first like any operator would.\n        if False:',
     "tests/test_backend.py -k sweep_needs"),
    ("Backend/python_Backend.py", "auto control steers a disarmed board",
     '        if not self.is_armed():\n            self._set_auto_control_status("waiting"',
     '        if False:\n            self._set_auto_control_status("waiting"', "tests/test_backend.py -k acts_only"),
    # Auto control safeguards
    ("Backend/python_Autonomy_Guard.py", "a NaN validation score is admitted",
     "        if not math.isfinite(score) or score < self.min_validation_r2:", "        if score < self.min_validation_r2:",
     "tests/test_autonomy_guard.py"),
    ("Backend/python_Autonomy_Guard.py", "no step limit",
     "        limited = np.clip(np.clip(proposed, current - step, current + step), low, high)",
     "        limited = np.clip(proposed, low, high)", "tests/test_autonomy_guard.py"),
    ("Backend/python_Autonomy_Guard.py", "a non-finite proposal is sent",
     "        if not np.all(np.isfinite(proposed)):", "        if False:", "tests/test_autonomy_guard.py"),
    ("Backend/python_Autonomy_Guard.py", "drift is judged on all history",
     "        recent = frame.tail(self.drift_window)", "        recent = frame", "tests/test_autonomy_guard.py"),
    ("Backend/python_Backend.py", "an unvalidated model drives the rig", "            if not admissible:", "            if False:",
     "tests/test_backend.py -k refused"),
    ("Backend/python_Backend.py", "drift does not hold the rig", "        if drifted:", "        if False:",
     "tests/test_backend.py -k training_data"),
    ("Backend/python_Backend.py", "the raw proposal is sent", "        self.set_pin_voltages(review.targets)",
     "        self.set_pin_voltages(proposal)", "tests/test_backend.py -k wild"),
    ("Backend/python_dashboard_script.py", "a manual edit leaves auto control on",
     '        if get_auto_control()["enabled"]:\n            set_auto_control(enabled=False)',
     '        if False:\n            set_auto_control(enabled=False)', "tests/test_dashboard.py -k manual"),
    # Security
    ("Backend/python_dashboard_script.py", "an observer may make the rig live",
     '    if role == "observer" and changed & OPERATOR_ONLY_INPUTS:', "    if False:", "tests/test_security.py -k observer"),
    ("Backend/python_dashboard_script.py", "operator actions are not audited",
     "    if changed & AUDITED_INPUTS:", "    if False:", "tests/test_security.py -k audited"),
    ("Backend/python_dashboard_script.py", "the dashboard serves off loopback without TLS",
     "    if not (tls_cert and tls_key) and not allow_plaintext:", "    if False:", "tests/test_security.py -k tls"),
    ("Backend/python_dashboard_script.py", "an observer may start a sweep",
     '    "sweep-btn.n_clicks", "save-model-button.n_clicks", "save-dataset-button.n_clicks",',
     '    "save-model-button.n_clicks", "save-dataset-button.n_clicks",', "tests/test_security.py -k operator_only"),
    ("Backend/python_Backend.py", "a dataset's hash is not of its contents",
     "            sha256=audit.file_sha256(csv_path),", '            sha256="0" * 64,', "tests/test_security.py -k dataset"),
    ("Backend/python_RNN_Controller.py", "a checkpoint loses its provenance",
     '"validation": dict(VALIDATION), "provenance": dict(PROVENANCE)}', '"validation": dict(VALIDATION)}',
     "tests/test_security.py -k checkpoint"),
    ("tools/generate_sbom.py", "SBOM package URLs are not encoded",
     "    purl = f\"pkg:{purl_type}/{quote(name, safe='/')}@{quote(version, safe='')}\"",
     '    purl = f"pkg:{purl_type}/{name}@{version}"', "tests/test_security.py -k sbom"),
    # The documents, against the code they describe
    ("../docs/protocol.md", "a command missing from the protocol document",
     "| `CLEAR LOG` | `ACK CLEAR LOG` | Clears the persisted history and the unexpected reset count |\n", "", "tests/test_docs.py"),
    ("../docs/protocol.md", "a fault's severity misstated", "| `GATE_MISMATCH` | Critical |", "| `GATE_MISMATCH` | Warning |",
     "tests/test_docs.py"),
    ("../docs/acceptance-test-procedure.md", "a bench step drops a requirement that names it",
     "Verifies BIT-01, BIT-02, BIT-03, LINK-06.", "Verifies BIT-01, BIT-02, BIT-03.", "tests/test_docs.py"),
    # The verification tools themselves
    ("tools/traceability.py", "an untested requirement goes unreported",
     '        if "Test" in requirement.methods and not any(not test.startswith("hil/") for test in requirement.tests):',
     "        if False:", "tests/test_traceability.py"),
    ("tools/check_firmware_size.py", "an oversized image passes",
     "    return share <= budget, f\"{build_dir.name}:", "    return True, f\"{build_dir.name}:", "tests/test_build_quality.py -k over"),
]

PROJECT_PARTS = ("include", "src", "tools", "platformio.ini", "Backend")
IGNORED = shutil.ignore_patterns("__pycache__", "*.pt", "models", "datasets", ".coverage", "*.jsonl", "sweep_dataset_*")


def copy_project(destination: Path):
    for part in PROJECT_PARTS:
        source = FIRMWARE_DIR / part
        if source.is_dir():
            shutil.copytree(source, destination / part, ignore=IGNORED)
        else:
            shutil.copy(source, destination / part)

    # Files the build-quality tests read from outside the firmware directory.
    workflows = destination.parent / ".github" / "workflows"
    workflows.mkdir(parents=True, exist_ok=True)
    shutil.copy(FIRMWARE_DIR.parent / ".github" / "workflows" / "ci.yml", workflows / "ci.yml")
    shutil.copytree(FIRMWARE_DIR.parent / "docs", destination.parent / "docs")


def run_selection(project: Path, selection: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", *selection.split()],
                          cwd=project / "Backend", capture_output=True, text=True, timeout=600)


@pytest.mark.parametrize("path, name, old, new, selection", BACKEND_CASES, ids=[case[1] for case in BACKEND_CASES])
def test_a_broken_backend_check_is_caught(tmp_path, path, name, old, new, selection):
    project = tmp_path / "repo" / "Embedded Controller"
    copy_project(project)

    target = project / path
    raw = target.read_bytes()
    bom = raw.startswith(b"\xef\xbb\xbf")
    text = raw.decode("utf-8-sig").replace("\r\n", "\n")
    target.write_bytes((b"\xef\xbb\xbf" if bom else b"") + mutate(text, old, new).encode("utf-8"))

    result = run_selection(project, selection)
    # 1 is "tests failed"; 5 would be "no tests selected", which proves nothing.
    assert result.returncode == 1, f"nothing noticed: {name}\n{result.stdout[-2000:]}{result.stderr[-1000:]}"


def test_the_unmutated_backend_passes_every_selection_used_above(tmp_path):
    project = tmp_path / "repo" / "Embedded Controller"
    copy_project(project)

    for selection in sorted({case[4] for case in BACKEND_CASES}):
        result = run_selection(project, selection)
        assert result.returncode == 0, f"{selection}\n{result.stdout[-2000:]}"
