"""Checks the hardware-in-the-loop suite itself, against the simulated board in hil/.

A rig test that has never run is a liability: a wrong reply string or a timing slip would
only show up with someone standing at the rig. So the whole suite runs here against a
board simulated at the protocol level, once as it should behave and once for each way
the simulator can be told to misbehave, where the rig test that guards that behaviour
must fail. None of this says anything about the real hardware.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]

pytestmark = [pytest.mark.mutation, pytest.mark.req("QA-04")]


def run_hil(selection=(), fault=""):
    environment = {**os.environ, "TESTBED_HIL_PORT": "sim", "TESTBED_HIL_ALLOW_ARM": "1", "TESTBED_SIM_FAULT": fault}
    environment.pop("TESTBED_SOAK_MINUTES", None)
    return subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "hil", *selection],
                          cwd=BACKEND_DIR, env=environment, capture_output=True, text=True, timeout=900)


def test_the_rig_suite_passes_against_a_board_that_behaves():
    result = run_hil()
    assert result.returncode == 0, result.stdout[-3000:]
    assert " failed" not in result.stdout


@pytest.mark.parametrize("fault, catcher", [
    ("ignores_crc", "test_a_corrupted_command_is_refused_and_counted"),
    ("late_failsafe", "test_the_failsafe_disarms_5_s_after_the_last_command"),
    ("energises_while_safe", "test_a_command_that_would_energise_an_output_is_refused_while_safe"),
    ("drops_readings", "test_readings_arrive_at_20_hz_numbered_and_without_gaps"),
    ("uncounted_overflow", "test_a_flood_of_garbage_is_discarded_and_the_board_keeps_answering"),
    ("blocking_serial", "test_the_backends_status_poll_does_not_hold_up_the_loop"),
])
def test_each_rig_test_fails_against_a_board_that_gets_its_one_thing_wrong(fault, catcher):
    result = run_hil(("-k", catcher), fault=fault)
    # 1 is "tests failed"; 5 would be "no tests selected", which proves nothing.
    assert result.returncode == 1, f"{catcher} passed against a board that {fault}:\n{result.stdout[-2000:]}"
