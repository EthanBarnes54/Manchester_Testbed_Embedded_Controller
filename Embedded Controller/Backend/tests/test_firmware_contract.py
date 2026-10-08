"""Contracts between the firmware and the backend, checked without a board.

Everything here is read straight out of main.cpp and platformio.ini, so the checks stay
honest as those files change instead of testing a hand-copied list that could drift.
"""

import ast
import re
from pathlib import Path

import pytest

from host_build import build_and_run, needs_compiler

BACKEND_DIR = Path(__file__).resolve().parents[1]
FIRMWARE_DIR = BACKEND_DIR.parent

MAIN_CPP = (FIRMWARE_DIR / "src" / "main.cpp").read_text(encoding="utf-8")
PLATFORMIO_INI = (FIRMWARE_DIR / "platformio.ini").read_text(encoding="utf-8")
# utf-8-sig because the backend source carries a byte order mark.
BACKEND_SOURCE = (BACKEND_DIR / "python_Backend.py").read_text(encoding="utf-8-sig")

ADS1115_FULL_SCALE_V = {"GAIN_TWOTHIRDS": 6.144, "GAIN_ONE": 4.096, "GAIN_TWO": 2.048}


def cpp_constant(name):
    match = re.search(rf"constexpr\s+[\w:\s]+?\s{name}\s*=\s*([^;]+);", MAIN_CPP)
    assert match, f"{name} not found in main.cpp"
    return match.group(1).strip()


def backend_constant(name):
    for node in ast.parse(BACKEND_SOURCE).body:
        if isinstance(node, ast.Assign) and any(getattr(target, "id", None) == name for target in node.targets):
            return ast.literal_eval(node.value)

    raise AssertionError(f"{name} not found in python_Backend.py")


def cpp_block(header):
    """Returns the brace-delimited body that follows the first match of header."""

    start = MAIN_CPP.index(header)
    opening = MAIN_CPP.index("{", start)
    depth = 0

    for index in range(opening, len(MAIN_CPP)):
        depth += {"{": 1, "}": -1}.get(MAIN_CPP[index], 0)

        if depth == 0:
            return MAIN_CPP[opening + 1 : index]

    raise AssertionError(f"Unbalanced braces after {header}")


def dispatch_branches():
    """The command chain in handle_command, in order, as [(label, [(kind, literal), ...])]."""

    body = cpp_block("void handle_command(String command)")
    branches = []

    for match in re.finditer(r"\bif\s*\(", body):
        depth, index = 1, match.end()

        while depth:
            depth += {"(": 1, ")": -1}.get(body[index], 0)
            index += 1

        condition = body[match.end() : index - 1]
        predicates = re.findall(r"command\.(equalsIgnoreCase|startsWith)\(\"([^\"]+)\"\)", condition)

        if predicates:
            branches.append((predicates[0][1], predicates))

    return branches


def route(command):
    """Which branch the firmware would run for this line, or None for 'Unknown command'."""

    command = command.strip()

    for label, predicates in dispatch_branches():
        for kind, literal in predicates:
            if kind == "equalsIgnoreCase" and command.lower() == literal.lower():
                return label

            if kind == "startsWith" and command.startswith(literal):
                return label

    return None


# Every command the backend and dashboard actually send, plus the operator ones.
@pytest.mark.parametrize(
    "command, branch",
    [
        ("PING", "PING"),
        ("ping", "PING"),
        ("READ", "READ"),
        ("PINS", "GET PINS"),
        ("GET PINS", "GET PINS"),
        ("PIN 3 512", "PIN"),
        ("PIN cone_1 100", "PIN"),
        ("TARGETS 1.000000 0.500000 0.000000 3.300000 2.000000", "TARGETS"),
        ("SWITCH_PERIOD_US 250", "SWITCH_PERIOD_US"),
        ("NONSENSE", None),
    ],
)
def test_every_command_reaches_its_own_handler(command, branch):
    assert route(command) == branch


def test_no_exact_command_is_shadowed_by_an_earlier_prefix():
    branches = dispatch_branches()

    for position, (label, predicates) in enumerate(branches):
        for kind, literal in predicates:
            if kind != "equalsIgnoreCase":
                continue

            for earlier_label, earlier_predicates in branches[:position]:
                for earlier_kind, earlier_literal in earlier_predicates:
                    assert not (earlier_kind == "startsWith" and literal.startswith(earlier_literal)), (
                        f"{literal!r} is swallowed by the earlier startsWith({earlier_literal!r}) branch"
                    )


@pytest.mark.req("SW-05")
def test_switch_bounds_match_the_backend():
    assert int(cpp_constant("SWITCH_PERIOD_MIN_US")) == backend_constant("SWITCH_PERIOD_MIN_US")
    assert int(cpp_constant("SWITCH_PERIOD_MAX_US")) == backend_constant("SWITCH_PERIOD_MAX_US")


def arduino_ledc_timer(channel):
    """The (group, timer) the Arduino core gives a channel: group c / 8, timer (c / 2) % 4."""

    return channel // 8, (channel // 2) % 4


@pytest.mark.req("SW-01")
def test_switch_has_an_ledc_timer_of_its_own():
    switch_channel = int(cpp_constant("SWITCH_LEDC_CHANNEL"))
    setpoint_channels = [int(c) for c in re.findall(r"\d+", cpp_constant(r"LED_CONTROL_CHANNELS\[CONTROLLED_PULSE_CHANNELS\]"))]

    assert len(setpoint_channels) == 5
    assert switch_channel not in setpoint_channels
    assert arduino_ledc_timer(switch_channel) not in {arduino_ledc_timer(c) for c in setpoint_channels}
    # The firmware derives its timer and output signal from the channel the same way.
    assert cpp_constant("SWITCH_LEDC_TIMER") == "static_cast<ledc_timer_t>((SWITCH_LEDC_CHANNEL / 2) % 4)"
    assert cpp_constant("SWITCH_LEDC_SIGNAL") == "LEDC_HS_SIG_OUT0_IDX + SWITCH_LEDC_CHANNEL"


@pytest.mark.req("SW-03", "SAF-01")
def test_switch_line_is_driven_low_before_anything_else_at_boot():
    first_statement = cpp_block("\nvoid setup()").strip().split(";")[0]
    assert first_statement == "SwitchLine::hold_low_at_boot()"


@pytest.mark.req("SW-04")
def test_the_ledc_handover_runs_out_of_line_from_iram():
    # Inlined into flash code, a cache miss between routing the pin and releasing the
    # count would stretch the first pulse by microseconds.
    assert "static void NOINLINE_ATTR IRAM_ATTR hand_pin_to_ledc()" in MAIN_CPP
    assert "hand_pin_to_ledc();" in cpp_block("bool start_hardware(unsigned long period_us)")


@pytest.mark.req("SAF-03", "SW-03")
def test_failsafe_holds_the_switch_line_low():
    assert "stop_switching(0);" in cpp_block("void engage_safe_state()")
    assert "switch_line_.hold(switch_level != 0);" in cpp_block("void stop_switching(int switch_level)")


def firmware_pins():
    return {name: int(number) for name, number in re.findall(r"constexpr int (\w+_PIN) = (\d+);", MAIN_CPP)}


@pytest.mark.req("SW-03", "SAF-01")
@pytest.mark.parametrize("name", ["SWITCH_ARMED_PIN", "HEARTBEAT_PIN"])
def test_safety_outputs_sit_on_free_pins_untouched_at_boot(name):
    pins = firmware_pins()
    pin = pins.pop(name)

    assert pin not in pins.values(), f"{name} shares a pin with another output"
    assert pin not in {0, 2, 5, 12, 15}, "strapping pins are read at reset"
    assert pin not in set(range(6, 12)) | {16, 17}, "flash, and the PSRAM probe's chip-select and clock"
    assert pin not in {1, 3, 21, 22}, "UART0 and the I2C bus to the ADS1115"
    assert pin < 32, "written through the GPIO 0-31 output registers"


@pytest.mark.req("SAF-03")
def test_the_failsafe_closes_the_gate_before_anything_else():
    body = cpp_block("void engage_safe_state()")
    assert body.strip().startswith("set_switch_armed(false);")
    assert body.index("set_switch_armed(false);") < body.index("stop_switching(0);")


@pytest.mark.req("SAF-02")
def test_the_gate_opens_only_on_an_explicit_arm():
    # The one place the gate is opened is SystemSupervisor::arm(), after the safety state agreed.
    assert MAIN_CPP.count("set_switch_armed(true)") == 1
    arm = cpp_block("  void arm()")
    assert arm.index("state_.arm_refusal()") < arm.index("state_.arm();") < arm.index("channels_.set_switch_armed(true);")

    # Setup leaves the board Safe, and a command arriving does not re-arm it.
    assert "set_switch_armed(true)" not in cpp_block("\nvoid setup()")
    assert "set_switch_armed(true)" not in cpp_block("void handle_command(String command)")


@pytest.mark.req("SAF-02")
@pytest.mark.parametrize(
    "header, guarded_before",
    [
        ('command.startsWith("TARGETS")', "channels_.apply_target_voltages(args)"),
        ('command.startsWith("PIN")', "channels_.set_channel(pin_index, value)"),
        ('command.startsWith("SWITCH_PERIOD_US")', "channels_.automate_switching("),
    ],
)
def test_commands_that_drive_an_output_are_refused_unless_armed(header, guarded_before):
    handler = cpp_block("void handle_command(String command)")
    branch = handler[handler.index(header):]
    branch = branch[: branch.index(guarded_before) + len(guarded_before)]

    assert "supervisor_.armed()" in branch
    assert '"ERROR: Not armed!"' in branch


@pytest.mark.req("SAF-02", "SAF-04")
@pytest.mark.parametrize("command", ["ARM", "DISARM", "FAULTS", "CLEAR FAULTS", "CLEAR LOG", "arm", "Clear Faults"])
def test_safety_commands_reach_their_own_handlers(command):
    assert route(command) == command.upper()


@pytest.mark.req("SAF-06", "SAF-07")
def test_a_hung_loop_resets_the_chip_and_an_update_never_runs_while_armed():
    assert "enableLoopWDT();" in cpp_block("\nvoid setup()")
    assert "ota_wifi_service.loop(!supervisor.armed());" in cpp_block("\nvoid loop()")
    assert "if (ota_started_ && updates_allowed)" in MAIN_CPP

    # The upload holds the loop without feeding the watchdog, so it stands down for it.
    begin = cpp_block("void begin(const char* ssid, const char* password, const char* hostname)")
    on_start = begin[begin.index("ArduinoOTA.onStart"):begin.index("ArduinoOTA.onEnd")]
    assert "disableLoopWDT();" in on_start
    assert "enableLoopWDT();" in begin[begin.index("ArduinoOTA.onEnd"):]


@pytest.mark.req("SAF-06")
def test_the_heartbeat_is_driven_every_pass():
    # First thing each pass, after only the pass timer starts.
    statements = [statement.strip() for statement in cpp_block("\nvoid loop()").split(";")]
    assert statements[:2] == ["const unsigned long pass_started_us = micros()", "supervisor.heartbeat()"]


def worst_case_reply_burst():
    """The most the board can owe the host at once, in characters: a keepalive's replies
    (OK, FAULTS, HEALTH) and VERSION, with every fault listed and counted and every number
    at its widest, plus two readings. Each line carries "*XXXX" and "\\r\\n"."""

    header = (FIRMWARE_DIR / "include" / "safety_state.h").read_text(encoding="utf-8")
    names = re.findall(r'return "([A-Z_]+)";', header[header.index("name_of(Fault"):header.index("name_of(Mode")])
    names.remove("UNKNOWN")
    every = ",".join(names)
    widest = "4294967295"

    faults = (f"FAULTS mode=FAULT active={every} latched={every} history={every} counts="
              + ",".join(f"{name}:65535" for name in names)
              + f" boots={widest} unexpected_resets={widest} last_reset=UNKNOWN")
    health_keys = re.findall(r'line \+= String\(" (\w+)="\)', cpp_block("  void report_health()"))
    health = "HEALTH mode=FAULT " + " ".join(f"{key}={widest}" for key in health_keys)
    version = "VERSION firmware=" + "x" * 40 + " protocol=2 build=esp32_deploy gate_loopback=1 setpoint_readback=1"
    reading = f"MEASURED -4.09600 V seq={widest} t_ms={widest}"

    return sum(len(line) + len("*XXXX\r\n") for line in ("OK", faults, health, version, reading, reading))


@pytest.mark.req("LINK-06")
def test_the_largest_burst_of_replies_fits_the_serial_transmit_buffer():
    burst = worst_case_reply_burst()
    assert burst > 128, "the burst fits the UART's FIFO alone; this check has stopped measuring anything"
    assert int(cpp_constant("SERIAL_TX_BUFFER_BYTES")) >= burst

    # The core only takes a buffer size before the UART starts.
    setup = cpp_block("\nvoid setup()")
    assert setup.index("Serial.setTxBufferSize(SERIAL_TX_BUFFER_BYTES);") < setup.index("Serial.begin(")


@pytest.mark.req("SAF-05")
def test_an_unexpected_reset_comes_up_in_fault():
    begin = cpp_block("  void begin() {\n    pinMode(HEARTBEAT_PIN")
    assert "reset_was_unexpected(reset_reason_)" in begin
    assert "raise(safety::Fault::UnexpectedReset);" in begin

    for reason in ("ESP_RST_PANIC", "ESP_RST_INT_WDT", "ESP_RST_TASK_WDT", "ESP_RST_WDT", "ESP_RST_BROWNOUT"):
        assert reason in cpp_block("static bool reset_was_unexpected(esp_reset_reason_t reason)")


@pytest.mark.req("LINK-04")
def test_keepalive_runs_well_inside_the_failsafe_timeout():
    timeout_s = int(cpp_constant("COMMAND_TIMEOUT_MS").rstrip("UL")) / 1000.0
    assert timeout_s >= 2 * backend_constant("KEEPALIVE_INTERVAL_SEC")


@pytest.mark.req("MEAS-03")
def test_adc_range_fits_the_signal_without_wasting_resolution():
    gain = cpp_constant("ADC_GAIN")
    assert "adc_.setGain(ADC_GAIN)" in cpp_block("bool begin()")

    full_scale = ADS1115_FULL_SCALE_V[gain]
    assert full_scale >= 3.3, "the range must still cover the 0-3.3 V signal"
    assert gain == min(
        (name for name, volts in ADS1115_FULL_SCALE_V.items() if volts >= 3.3), key=ADS1115_FULL_SCALE_V.get
    ), "a tighter gain would still cover 3.3 V"


@pytest.mark.req("MEAS-03")
def test_measured_lines_keep_sub_count_precision():
    count_v = ADS1115_FULL_SCALE_V[cpp_constant("ADC_GAIN")] / 32768
    places = int(cpp_constant("MEASURED_DECIMAL_PLACES"))
    assert 10 ** -places < count_v

    # One formatter for every MEASURED line, and it must pass the precision explicitly.
    assert MAIN_CPP.count('"MEASURED "') == 1
    assert "String(volts, MEASURED_DECIMAL_PLACES)" in cpp_block("void report_voltage(float volts)")


@pytest.mark.req("MEAS-01", "BIT-02")
@pytest.mark.parametrize(
    "header",
    [
        "class SwitchLine",
        "class ChannelController",
        "class MeasurementService",
        "class LedIndicator",
        "class OtaWifiService",
        "class SystemSupervisor",
        "class CommandProcessor",
        "void enforce_command_timeout",
    ],
)
def test_nothing_on_the_loop_path_blocks(header):
    assert "delay(" not in cpp_block(header)


@pytest.mark.req("MEAS-01")
def test_loop_only_yields():
    # Anchored to the line start so it finds the free loop(), not OtaWifiService::loop.
    assert re.findall(r"delay\((\d+)\)", cpp_block("\nvoid loop()")) == ["1"]


@pytest.mark.req("SEC-08")
def test_platform_and_libraries_are_pinned_exactly():
    platform = re.search(r"^platform\s*=\s*(\S+)", PLATFORMIO_INI, re.M).group(1)
    assert re.fullmatch(r"espressif32@\d+\.\d+\.\d+", platform), platform
    assert len(re.findall(r"^platform\s*=", PLATFORMIO_INI, re.M)) == 1, "every env should share the one pin"

    libraries = re.findall(r"^\s+([\w-]+/[^@\n]+@\S+)\s*$", PLATFORMIO_INI, re.M)
    assert libraries, "no lib_deps found"

    for library in libraries:
        assert re.search(r"@\d+\.\d+\.\d+$", library), f"{library} is not pinned to an exact version"


@pytest.mark.req("SEC-08")
def test_no_machine_specific_upload_port():
    for value in re.findall(r"^upload_port\s*=\s*(.+)$", PLATFORMIO_INI, re.M):
        assert value.strip() == "${sysenv.TESTBED_SERIAL_PORT}"


LED_HARNESS = r"""
#include <cstdio>
#include <cstdlib>
#define HIGH 1
#define LOW 0
#define OUTPUT 1
static unsigned long now_ms = 0;
static int led = -1;
unsigned long millis() { return now_ms; }
void pinMode(int, int) {}
void digitalWrite(int, int value) { led = value; }
void delay(unsigned long) { std::puts("delay() called"); std::abort(); }
constexpr unsigned long HEARTBEAT_INTERVAL_MS = 2000;
@@CLASS@@
static int failures = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d at %lu ms\n", __LINE__, now_ms); ++failures; } } while (0)
static void run_to(LedIndicator& l, unsigned long t, bool beat) {
  while (now_ms < t) { ++now_ms; if (beat) l.heartbeat(now_ms); l.update(now_ms); }
}
int main() {
  { LedIndicator l(2); l.begin(); now_ms = 0; l.set_low();
    l.blink_error(2, 70);
    CHECK(led == HIGH && l.busy());
    run_to(l, 69, false);  CHECK(led == HIGH);
    run_to(l, 70, false);  CHECK(led == LOW);
    run_to(l, 140, false); CHECK(led == HIGH);
    run_to(l, 210, false); CHECK(led == LOW);
    run_to(l, 280, false); CHECK(!l.busy()); }
  { LedIndicator l(2); l.begin(); now_ms = 0; l.set_low();
    int flashes = 0, previous = LOW; unsigned long lit = 0;
    while (now_ms < 10000) { ++now_ms; l.heartbeat(now_ms); l.update(now_ms);
      if (led == HIGH) { ++lit; }
      if (led == HIGH && previous == LOW) { ++flashes; }
      previous = led; }
    CHECK(flashes == 4); CHECK(lit == 200); }
  { LedIndicator l(2); l.begin(); now_ms = 0; l.blink_error(5); l.set_low();
    CHECK(led == HIGH && l.busy()); }
  return failures ? 1 : 0;
}
"""


def led_harness_source(main_cpp=None):
    led_class = re.search(r"^class LedIndicator \{.*?^\};", main_cpp or MAIN_CPP, re.S | re.M).group(0)
    return LED_HARNESS.replace("@@CLASS@@", led_class)


@needs_compiler
def test_led_indicator_on_the_host(tmp_path):
    run = build_and_run(led_harness_source(), tmp_path, "led")
    assert run.returncode == 0, run.stdout + run.stderr


# ----------------------------------------------------------------------------
#                               Built-in test
# ----------------------------------------------------------------------------


@pytest.mark.req("BIT-03")
@pytest.mark.parametrize("command", ["SELFTEST", "HEALTH", "selftest"])
def test_self_test_and_health_reach_their_own_handlers(command):
    assert route(command) == command.upper()


@pytest.mark.req("BIT-01")
def test_the_power_on_self_test_runs_after_everything_it_checks_is_up():
    setup = cpp_block("\nvoid setup()")
    assert setup.index("measurement_service.begin()") < setup.index("channels.begin();") < setup.index("supervisor.begin();")
    assert "self_test();" in cpp_block("  void begin() {\n    pinMode(HEARTBEAT_PIN")


@pytest.mark.req("BIT-01")
def test_the_self_test_checks_what_the_outputs_depend_on():
    body = cpp_block("  void self_test()")
    for check, fault in [("clocks_as_designed()", "ClockConfig"), ("switch_generators_ready()", "SwitchGenerator"),
                         ("measurement_.lost()", "AdcLost"), ("memory_above_floor()", "LowMemory")]:
        assert check in body and f"safety::Fault::{fault}" in body

    clocks = cpp_block("static bool clocks_as_designed()")
    assert "getApbFrequency() == APB_CLK_FREQ" in clocks
    assert "APB_CTRL_PLL_TICK_NUM" in clocks and "SWITCH_LEDC_CLOCK_HZ" in clocks


@pytest.mark.req("BIT-02")
def test_every_loop_pass_is_timed_and_monitored():
    loop = cpp_block("\nvoid loop()")
    assert loop.strip().startswith("const unsigned long pass_started_us = micros();")
    assert "supervisor.monitor(now_ms, static_cast<uint32_t>(micros() - pass_started_us));" in loop
    assert loop.index("supervisor.monitor(") < loop.index("delay(1);")


@pytest.mark.req("BIT-02")
def test_continuous_checks_feed_their_faults():
    monitor = cpp_block("void monitor(unsigned long now_ms, uint32_t pass_us)")
    assert "loop_timing_.record(pass_us);" in monitor
    assert "set_fault(safety::Fault::AdcLost, measurement_.lost());" in monitor
    assert "safety::Fault::LowMemory" in monitor and "safety::Fault::LoopOverrun" in monitor

    assert "supervisor_.note_serial_overflow();" in cpp_block("void poll_serial()")


@pytest.mark.req("BIT-02")
def test_the_loop_budget_leaves_room_for_the_longest_legitimate_pass():
    # The longest pass by design is a period change priming the LEDC: two periods plus slack.
    longest_priming_us = 4 * int(cpp_constant("SWITCH_HARDWARE_MAX_US")) + int(cpp_constant("SWITCH_LEDC_SETTLE_MARGIN_US"))
    assert int(cpp_constant("LOOP_BUDGET_US")) >= 4 * longest_priming_us


# ----------------------------------------------------------------------------
#                            Output verification
# ----------------------------------------------------------------------------


@pytest.mark.req("OUT-03")
def test_verification_hardware_is_off_unless_a_build_declares_it():
    for flag in ("TESTBED_GATE_LOOPBACK", "TESTBED_SETPOINT_READBACK"):
        assert f"#ifndef {flag}\n#define {flag} 0\n#endif" in MAIN_CPP

    # Both checks are written as ordinary branches on constants, so the compiler checks
    # them in every build, not only the one that turns them on.
    assert "#if TESTBED_GATE_LOOPBACK" not in MAIN_CPP and "#if TESTBED_SETPOINT_READBACK" not in MAIN_CPP


@pytest.mark.req("OUT-01")
def test_the_loopback_pin_is_input_only_and_free():
    pins = firmware_pins()
    loopback = pins.pop("GATE_LOOPBACK_PIN")

    assert 34 <= loopback <= 39, "an input-only pin, which nothing at boot can drive"
    assert loopback not in pins.values()


@pytest.mark.req("OUT-01", "OUT-02", "BIT-01")
def test_outputs_are_verified_every_pass_and_at_power_on():
    monitor = cpp_block("void monitor(unsigned long now_ms, uint32_t pass_us)")
    assert monitor.index("outputs_.check(state_.armed());") < monitor.index("report_output_faults();")

    report = cpp_block("void report_output_faults()")
    for fault, verdict in [("GateMismatch", "gate_fault()"), ("SwitchFrequency", "frequency_fault()"),
                           ("SetpointMismatch", "readback_fault()")]:
        assert f"set_fault(safety::Fault::{fault}, outputs_.{verdict});" in report

    self_test = cpp_block("  void self_test()")
    assert "report_output_faults();" in self_test
    assert "outputs_.gate_verdict()" in self_test and "outputs_.readback_verdict()" in self_test

    setup = cpp_block("\nvoid setup()")
    assert setup.index("output_verifier.begin();") < setup.index("supervisor.begin();")


@pytest.mark.req("OUT-02", "MEAS-01")
def test_a_readback_conversion_always_ends_before_the_next_diode_one():
    assert cpp_constant("READBACK_START_WINDOW_MS") == "MEASUREMENT_INTERVAL_MS - ADC_CONVERSION_TIMEOUT_MS"
    window = int(cpp_constant("MEASUREMENT_INTERVAL_MS").rstrip("UL")) - int(cpp_constant("ADC_CONVERSION_TIMEOUT_MS").rstrip("UL"))
    assert window > int(cpp_constant("ADC_FIRST_POLL_MS").rstrip("UL")), "a diode conversion must be able to finish inside it"


# ----------------------------------------------------------------------------
#                              Link integrity
# ----------------------------------------------------------------------------


@pytest.mark.req("LINK-02")
def test_firmware_and_backend_speak_the_same_protocol_version():
    assert int(cpp_constant("PROTOCOL_VERSION")) == backend_constant("PROTOCOL_VERSION")


@pytest.mark.req("LINK-01")
def test_every_line_to_the_host_carries_a_crc():
    # The only direct writes are send_line() itself and the OTA progress counter.
    writes = re.findall(r"Serial\.(?:print|println|printf|write)\(", MAIN_CPP)
    assert len(writes) == 3

    sender = cpp_block("void send_line(const String& line)")
    assert "line_protocol::crc16(line.c_str(), line.length())" in sender
    assert "Serial.print(line);" in sender and "Serial.println(suffix);" in sender


@pytest.mark.req("LINK-01", "SAF-03")
def test_a_command_with_a_bad_crc_is_refused_and_does_not_feed_the_failsafe():
    handler = cpp_block("void handle_command(String command)")
    check = handler.index("line_protocol::Check::Invalid")

    assert check < handler.index("last_command_ms_ = millis();")
    assert check < handler.index('command.equalsIgnoreCase("PING")')
    assert '"ERROR: Bad checksum!"' in handler and "supervisor_.note_bad_checksum();" in handler


@pytest.mark.req("MEAS-02")
def test_every_reading_is_numbered_and_timestamped():
    formatter = cpp_block("void report_voltage(float volts)")
    assert '" V seq=" + sequence_ + " t_ms=" + millis()' in formatter
    assert "++sequence_;" in formatter


@pytest.mark.req("LINK-02", "OUT-03")
def test_the_version_reply_names_the_build_and_its_hardware():
    assert route("VERSION") == "VERSION"
    version = cpp_block("static void report_version()")
    for field in ("firmware=", "protocol=", "build=", "gate_loopback=", "setpoint_readback="):
        assert field in version

    assert re.search(r"^extra_scripts = post:tools/firmware_version.py$", PLATFORMIO_INI, re.M)
    assert (FIRMWARE_DIR / "tools" / "firmware_version.py").is_file()
