import random
import statistics
import time

import pytest

from rig import PROTOCOL_PREFIXES, fields, frame

pytestmark = pytest.mark.hil

PROTOCOL_VERSION = "2"
FAILSAFE_TIMEOUT_S = 5.0
# One loop pass (budget 20 ms) plus USB serial latency either way.
FAILSAFE_MARGIN_S = 0.25
MEASUREMENT_INTERVAL_MS = 50


@pytest.mark.req("LINK-02")
def test_the_board_speaks_this_protocol(rig):
    version = rig.version()
    assert version["protocol"] == PROTOCOL_VERSION, version
    assert version["firmware"] and version["build"]


@pytest.mark.req("LINK-01")
def test_every_protocol_line_from_the_board_carries_a_valid_crc(rig):
    since = rig.mark()

    with rig.keepalive():
        rig.faults()
        rig.health()
        time.sleep(3)

    protocol_lines = [line for line in rig.lines(since) if line.text.startswith(PROTOCOL_PREFIXES)]
    assert len(protocol_lines) > 40
    assert [line.text for line in protocol_lines if line.verdict != "valid"] == []


@pytest.mark.req("SAF-01", "BIT-01", "SW-03")
def test_after_a_reset_the_board_is_safe_and_its_self_test_passes(rig):
    rig.reset()
    faults = rig.faults()

    assert faults["mode"] == "SAFE", faults
    assert faults["last_reset"] in ("POWERON", "EXTERNAL"), faults
    assert rig.selftest().startswith("SELFTEST PASS")


@pytest.mark.req("BIT-03")
def test_the_self_test_and_health_report_answer_on_demand(safe_rig):
    assert safe_rig.selftest().startswith("SELFTEST PASS")
    health = safe_rig.health()
    assert {"loop_max_us", "loop_peak_us", "heap_free", "stack_free", "adc"} <= set(health)


@pytest.mark.req("SAF-02")
@pytest.mark.parametrize("command", ["TARGETS 1 0 0 0 0", "PIN squeeze_plate 100", "PIN switch_logic 1", "SWITCH_PERIOD_US 1000"])
def test_a_command_that_would_energise_an_output_is_refused_while_safe(safe_rig, command):
    assert safe_rig.command(command, ("ACK",)) == "ERROR: Not armed!"
    assert safe_rig.mode() == "SAFE"


@pytest.mark.req("SAF-02")
@pytest.mark.parametrize("command, reply", [("TARGETS 0 0 0 0 0", "ACK TARGETS"), ("SWITCH_PERIOD_US 0", "ACK SWITCH_PERIOD_US 0"),
                                            ("PIN squeeze_plate 0", "ACK PIN")])
def test_a_command_that_zeroes_an_output_is_always_accepted(safe_rig, command, reply):
    # Matched on the full reply: TARGETS is answered by an ACK PIN per channel first.
    assert safe_rig.command(command, (reply,)).startswith(reply)


@pytest.mark.req("LINK-01")
def test_a_corrupted_command_is_refused_and_counted(safe_rig):
    before = int(safe_rig.health()["bad_checksums"])

    assert safe_rig.command("PING*0000", ("OK",), framed=False) == "ERROR: Bad checksum!"
    # ARM with one digit of its CRC changed: refused, and the board stays SAFE.
    framed = frame("ARM")
    corrupted = framed[:-1] + ("0" if framed[-1] != "0" else "1")
    assert safe_rig.command(corrupted, ("ACK",), framed=False) == "ERROR: Bad checksum!"

    assert safe_rig.mode() == "SAFE"
    assert int(safe_rig.health()["bad_checksums"]) == before + 2


@pytest.mark.req("SAF-03", "LINK-01")
def test_corrupted_commands_do_not_keep_the_failsafe_away(safe_rig):
    safe_rig.command("PING", ("OK",))
    last_valid = safe_rig.last_sent_at
    since = safe_rig.mark()

    # A link that only delivers garbage must still trip the failsafe on time.
    for _ in range(int(FAILSAFE_TIMEOUT_S) + 2):
        safe_rig.send("PING*0000", framed=False)
        tripped = safe_rig.wait_for(lambda line: line.text.startswith("FAILSAFE outputs zeroed"), since, 1.0)
        if tripped:
            break

    assert tripped, "no FAILSAFE line while the host sent only corrupted commands"
    assert tripped.time - last_valid <= FAILSAFE_TIMEOUT_S + FAILSAFE_MARGIN_S

    recovering = safe_rig.mark()
    safe_rig.send("PING")
    assert safe_rig.wait_for(lambda line: line.text == "OK", recovering, 1.0)
    assert safe_rig.wait_for(lambda line: line.text.startswith("FAILSAFE cleared"), recovering, 1.0)


@pytest.mark.req("SAF-02", "SAF-04")
def test_arm_and_disarm(armable_rig):
    assert armable_rig.command("ARM", ("ACK ARM",)) == "ACK ARM"
    assert armable_rig.mode() == "ARMED"
    assert armable_rig.command("DISARM", ("ACK DISARM",)) == "ACK DISARM"
    assert armable_rig.mode() == "SAFE"


@pytest.mark.req("SAF-03")
def test_the_failsafe_disarms_5_s_after_the_last_command_and_the_host_returning_does_not_rearm(armable_rig):
    assert armable_rig.command("ARM", ("ACK ARM",)) == "ACK ARM"
    last_command = armable_rig.last_sent_at
    since = armable_rig.mark()

    tripped = armable_rig.wait_for(lambda line: line.text.startswith("FAILSAFE outputs zeroed"), since,
                                   FAILSAFE_TIMEOUT_S + 2)

    assert tripped, "the failsafe never tripped"
    elapsed = tripped.time - last_command
    assert FAILSAFE_TIMEOUT_S - 0.05 <= elapsed <= FAILSAFE_TIMEOUT_S + FAILSAFE_MARGIN_S, f"tripped after {elapsed:.3f} s"

    faults = armable_rig.faults()
    assert faults["mode"] == "SAFE", faults
    assert "HOST_TIMEOUT" in faults["history"]


@pytest.mark.req("SW-01", "SW-02")
def test_every_switch_time_in_range_is_accepted_and_one_outside_is_refused(armable_rig):
    armable_rig.command("ARM", ("ACK ARM",))

    for period_us in (1, 2, 5, 1000, 1001, 50_000, 2_000_000):
        assert armable_rig.command(f"SWITCH_PERIOD_US {period_us}", ("ACK SWITCH_PERIOD_US",)) == f"ACK SWITCH_PERIOD_US {period_us}"

    assert armable_rig.command("SWITCH_PERIOD_US 2000001", ("ACK SWITCH_PERIOD_US",)) == "ERROR: Switch time out of bounds!"
    assert armable_rig.mode() == "ARMED"


@pytest.mark.req("OUT-01", "SW-01", "SW-04")
@pytest.mark.parametrize("period_us", [1, 5, 100, 1000, 1001, 20_000])
def test_with_the_loopback_fitted_the_gate_switches_at_the_commanded_rate(armable_rig, period_us):
    if armable_rig.version().get("gate_loopback") != "1":
        pytest.skip("this build does not expect the gate loopback (GPIO 34)")

    armable_rig.command("ARM", ("ACK ARM",))

    with armable_rig.keepalive():
        assert armable_rig.command(f"SWITCH_PERIOD_US {period_us}", ("ACK SWITCH_PERIOD_US",)).endswith(str(period_us))
        # Long enough for several verification windows at the slowest period here.
        time.sleep(max(1.5, 6 * period_us / 1e6))
        health, faults = armable_rig.health(), armable_rig.faults()

    assert health["gate"] == "ok", health
    assert int(health["gate_edges"]) > 0, health
    assert not {"GATE_MISMATCH", "SWITCH_FREQUENCY"} & set(faults["active"].split(",")), faults
    assert faults["mode"] == "ARMED", faults


@pytest.mark.req("MEAS-01", "MEAS-02")
def test_readings_arrive_at_20_hz_numbered_and_without_gaps(safe_rig):
    since = safe_rig.mark()

    with safe_rig.keepalive():
        time.sleep(10)

    readings = safe_rig.lines(since, "MEASURED")
    numbered = [(line.time, fields(line.text)) for line in readings]
    sequence = [int(f["seq"]) for _, f in numbered]
    board_ms = [int(f["t_ms"]) for _, f in numbered]

    assert len(readings) >= 190, f"{len(readings)} readings in 10 s"
    assert sequence == list(range(sequence[0], sequence[0] + len(sequence))), "readings went missing"
    assert abs(statistics.mean(b - a for a, b in zip(board_ms, board_ms[1:])) - MEASUREMENT_INTERVAL_MS) <= 1
    host_rate = (len(readings) - 1) / (numbered[-1][0] - numbered[0][0])
    assert 19 <= host_rate <= 21, f"{host_rate:.2f} readings per second at the host"


@pytest.mark.req("LINK-06", "BIT-02")
def test_the_backends_status_poll_does_not_hold_up_the_loop(safe_rig):
    poll = b"".join((frame(command) + "\n").encode("utf-8") for command in ("PING", "FAULTS", "HEALTH"))
    before = int(safe_rig.health()["overruns"])

    for _ in range(5):
        since = safe_rig.mark()
        safe_rig.send_raw(poll)
        assert safe_rig.wait_for(lambda line: line.text.startswith("HEALTH "), since, 3.0)
        time.sleep(0.5)

    after = safe_rig.health()
    assert int(after["overruns"]) == before, f"{int(after['overruns']) - before} loop overruns during five polls"


@pytest.mark.req("BIT-02")
def test_the_loop_stays_inside_its_budget_and_memory_holds(safe_rig):
    with safe_rig.keepalive():
        start = safe_rig.health()
        time.sleep(10)
        end = safe_rig.health()

    assert int(end["overruns"]) == int(start["overruns"]) == 0, end
    assert int(end["loop_peak_us"]) < int(end["loop_budget_us"]), end
    assert int(end["heap_min"]) >= 32768 and int(end["stack_free"]) >= 1024, end
    assert end["adc"] == "ok" and int(end["adc_timeouts"]) == int(start["adc_timeouts"]), end
    assert int(end["heap_free"]) >= int(start["heap_free"]) - 1024, "free heap fell over 10 s"


@pytest.mark.req("LINK-05")
def test_a_flood_of_garbage_is_discarded_and_the_board_keeps_answering(safe_rig):
    before_health, before_faults = safe_rig.health(), safe_rig.faults()
    rng = random.Random(1)

    safe_rig.send_raw(b"X" * 4000 + b"\n")
    safe_rig.send_raw(b"".join(bytes(rng.randrange(32, 127) for _ in range(rng.randrange(1, 300))) + b"\n" for _ in range(200)))

    since = safe_rig.mark()
    safe_rig.send("VERSION")
    assert safe_rig.wait_for(lambda line: line.text.startswith("VERSION "), since, 5.0), "the board stopped answering"
    after_health, after_faults = safe_rig.health(), safe_rig.faults()

    assert int(after_health["rx_overflows"]) > int(before_health["rx_overflows"])
    assert after_faults["boots"] == before_faults["boots"], "the board restarted"
    assert after_faults["mode"] == "SAFE", after_faults
