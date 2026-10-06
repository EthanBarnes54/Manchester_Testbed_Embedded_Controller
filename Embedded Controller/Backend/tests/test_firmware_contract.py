"""Contracts between the firmware and the backend, checked without a board.

Everything here is read straight out of main.cpp and platformio.ini, so the checks stay
honest as those files change instead of testing a hand-copied list that could drift.
"""

import ast
import re
import shutil
import subprocess
from pathlib import Path

import pytest

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


def test_switch_bounds_match_the_backend():
    assert int(cpp_constant("SWITCH_PERIOD_MIN_US")) == backend_constant("SWITCH_PERIOD_MIN_US")
    assert int(cpp_constant("SWITCH_PERIOD_MAX_US")) == backend_constant("SWITCH_PERIOD_MAX_US")


def arduino_ledc_timer(channel):
    """The (group, timer) the Arduino core gives a channel: group c / 8, timer (c / 2) % 4."""

    return channel // 8, (channel // 2) % 4


def test_switch_has_an_ledc_timer_of_its_own():
    switch_channel = int(cpp_constant("SWITCH_LEDC_CHANNEL"))
    setpoint_channels = [int(c) for c in re.findall(r"\d+", cpp_constant(r"LED_CONTROL_CHANNELS\[CONTROLLED_PULSE_CHANNELS\]"))]

    assert len(setpoint_channels) == 5
    assert switch_channel not in setpoint_channels
    assert arduino_ledc_timer(switch_channel) not in {arduino_ledc_timer(c) for c in setpoint_channels}
    # The firmware derives its timer and output signal from the channel the same way.
    assert cpp_constant("SWITCH_LEDC_TIMER") == "static_cast<ledc_timer_t>((SWITCH_LEDC_CHANNEL / 2) % 4)"
    assert cpp_constant("SWITCH_LEDC_SIGNAL") == "LEDC_HS_SIG_OUT0_IDX + SWITCH_LEDC_CHANNEL"


def test_switch_line_is_driven_low_before_anything_else_at_boot():
    first_statement = cpp_block("\nvoid setup()").strip().split(";")[0]
    assert first_statement == "SwitchLine::hold_low_at_boot()"


def test_the_ledc_handover_runs_out_of_line_from_iram():
    # Inlined into flash code, a cache miss between routing the pin and releasing the
    # count would stretch the first pulse by microseconds.
    assert "static void NOINLINE_ATTR IRAM_ATTR hand_pin_to_ledc()" in MAIN_CPP
    assert "hand_pin_to_ledc();" in cpp_block("bool start_hardware(unsigned long period_us)")


def test_failsafe_holds_the_switch_line_low():
    assert "stop_switching(0);" in cpp_block("void engage_safe_state()")
    assert "switch_line_.hold(switch_level != 0);" in cpp_block("void stop_switching(int switch_level)")


def test_keepalive_runs_well_inside_the_failsafe_timeout():
    timeout_s = int(cpp_constant("COMMAND_TIMEOUT_MS").rstrip("UL")) / 1000.0
    assert timeout_s >= 2 * backend_constant("KEEPALIVE_INTERVAL_SEC")


def test_adc_range_fits_the_signal_without_wasting_resolution():
    gain = cpp_constant("ADC_GAIN")
    assert "adc_.setGain(ADC_GAIN)" in cpp_block("bool begin()")

    full_scale = ADS1115_FULL_SCALE_V[gain]
    assert full_scale >= 3.3, "the range must still cover the 0-3.3 V signal"
    assert gain == min(
        (name for name, volts in ADS1115_FULL_SCALE_V.items() if volts >= 3.3), key=ADS1115_FULL_SCALE_V.get
    ), "a tighter gain would still cover 3.3 V"


def test_measured_lines_keep_sub_count_precision():
    count_v = ADS1115_FULL_SCALE_V[cpp_constant("ADC_GAIN")] / 32768
    places = int(cpp_constant("MEASURED_DECIMAL_PLACES"))
    assert 10 ** -places < count_v

    # One formatter for every MEASURED line, and it must pass the precision explicitly.
    assert MAIN_CPP.count('"MEASURED "') == 1
    assert "String(volts, MEASURED_DECIMAL_PLACES)" in cpp_block("void report_voltage(float volts)")


@pytest.mark.parametrize(
    "header",
    [
        "class SwitchLine",
        "class ChannelController",
        "class MeasurementService",
        "class LedIndicator",
        "class OtaWifiService",
        "class CommandProcessor",
        "void enforce_command_timeout",
    ],
)
def test_nothing_on_the_loop_path_blocks(header):
    assert "delay(" not in cpp_block(header)


def test_loop_only_yields():
    # Anchored to the line start so it finds the free loop(), not OtaWifiService::loop.
    assert re.findall(r"delay\((\d+)\)", cpp_block("\nvoid loop()")) == ["1"]


def test_platform_and_libraries_are_pinned_exactly():
    platform = re.search(r"^platform\s*=\s*(\S+)", PLATFORMIO_INI, re.M).group(1)
    assert re.fullmatch(r"espressif32@\d+\.\d+\.\d+", platform), platform
    assert len(re.findall(r"^platform\s*=", PLATFORMIO_INI, re.M)) == 1, "every env should share the one pin"

    libraries = re.findall(r"^\s+([\w-]+/[^@\n]+@\S+)\s*$", PLATFORMIO_INI, re.M)
    assert libraries, "no lib_deps found"

    for library in libraries:
        assert re.search(r"@\d+\.\d+\.\d+$", library), f"{library} is not pinned to an exact version"


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


@pytest.mark.skipif(shutil.which("g++") is None, reason="needs a host C++ compiler")
def test_led_indicator_on_the_host(tmp_path):
    led_class = re.search(r"^class LedIndicator \{.*?^\};", MAIN_CPP, re.S | re.M).group(0)
    source = tmp_path / "led.cpp"
    binary = tmp_path / "led"
    source.write_text(LED_HARNESS.replace("@@CLASS@@", led_class))

    build = subprocess.run(["g++", "-std=c++17", "-Wall", "-o", str(binary), str(source)], capture_output=True, text=True)
    assert build.returncode == 0, build.stderr

    run = subprocess.run([str(binary)], capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr
