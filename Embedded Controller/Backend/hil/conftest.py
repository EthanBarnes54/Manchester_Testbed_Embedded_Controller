"""Fixtures for the hardware-in-the-loop tests. See README.md in this directory first.

Nothing here runs unless TESTBED_HIL_PORT names the board's serial port, and no test arms
the board unless TESTBED_HIL_ALLOW_ARM=1 as well: arming lets the switch line and the
setpoints reach whatever the rig has connected to them.
"""

import os

import pytest

from rig import Rig

PORT = os.environ.get("TESTBED_HIL_PORT", "")
ALLOW_ARM = os.environ.get("TESTBED_HIL_ALLOW_ARM", "") == "1"
RESET = os.environ.get("TESTBED_HIL_RESET", "1") != "0"


def pytest_collection_modifyitems(config, items):
    if PORT:
        return

    skip = pytest.mark.skip(reason="set TESTBED_HIL_PORT to the board's serial port to run the rig tests")
    for item in items:
        item.add_marker(skip)


@pytest.fixture(scope="session")
def rig():
    connection = Rig(PORT, reset=RESET)

    try:
        yield connection
    finally:
        # Whatever a test did, the board is left disarmed with its outputs at zero.
        try:
            connection.make_safe()
        finally:
            connection.close()


@pytest.fixture
def safe_rig(rig):
    """The rig, disarmed with every output at zero, before and after the test."""

    rig.make_safe()
    yield rig
    rig.make_safe()


@pytest.fixture
def armable_rig(safe_rig):
    """The rig, for a test that must arm it. Skipped unless the operator has said the rig
    is safe to energise."""

    if not ALLOW_ARM:
        pytest.skip("arms the board: set TESTBED_HIL_ALLOW_ARM=1 once the rig is safe to energise")

    if safe_rig.mode() == "FAULT":
        pytest.fail(f"the board is in FAULT before the test: {safe_rig.faults()}")

    return safe_rig
