import os
import time

import pytest

from rig import fields

SOAK_MINUTES = float(os.environ.get("TESTBED_SOAK_MINUTES", "0") or 0)
SOAK_SWITCH_US = int(os.environ.get("TESTBED_SOAK_SWITCH_US", "0") or 0)
ALLOW_ARM = os.environ.get("TESTBED_HIL_ALLOW_ARM", "") == "1"
HEALTH_EVERY_S = 60

pytestmark = [pytest.mark.hil, pytest.mark.soak,
              pytest.mark.skipif(SOAK_MINUTES <= 0, reason="set TESTBED_SOAK_MINUTES to run the soak test")]


@pytest.mark.req("QA-05", "MEAS-02", "BIT-02")
def test_the_board_runs_the_soak_period_without_a_fault_a_gap_or_an_overrun(safe_rig):
    switching = SOAK_SWITCH_US > 0 and ALLOW_ARM

    if switching:
        assert safe_rig.command("ARM", ("ACK ARM",)) == "ACK ARM"
        assert safe_rig.command(f"SWITCH_PERIOD_US {SOAK_SWITCH_US}", ("ACK SWITCH_PERIOD_US",)).endswith(str(SOAK_SWITCH_US))

    start_faults, since = safe_rig.faults(), safe_rig.mark()
    reports = []
    deadline = time.monotonic() + SOAK_MINUTES * 60

    with safe_rig.keepalive():
        reports.append(safe_rig.health())

        while time.monotonic() < deadline:
            time.sleep(min(HEALTH_EVERY_S, max(0.0, deadline - time.monotonic())))
            reports.append(safe_rig.health())

        end_faults = safe_rig.faults()

    readings = [int(fields(line.text)["seq"]) for line in safe_rig.lines(since, "MEASURED")]
    fault_lines = [line.text for line in safe_rig.lines(since, "FAULT ")]
    expected_mode = "ARMED" if switching else "SAFE"

    assert fault_lines == [], fault_lines
    assert readings == list(range(readings[0], readings[0] + len(readings))), "readings went missing"
    assert len(readings) >= SOAK_MINUTES * 60 * 19
    assert end_faults["boots"] == start_faults["boots"], "the board restarted"
    assert end_faults["mode"] == expected_mode, end_faults
    # The overrun count runs from boot, so it is held, not required to be zero.
    assert len({report["overruns"] for report in reports}) == 1, [report["overruns"] for report in reports]
    assert all(report["adc"] == "ok" for report in reports)
    # A leak shows as free heap that keeps falling; allow a little for fragmentation.
    assert int(reports[-1]["heap_free"]) >= int(reports[0]["heap_free"]) - 2048, [report["heap_free"] for report in reports]
    if switching:
        assert all(report["gate"] in ("ok", "off") for report in reports), [report["gate"] for report in reports]
