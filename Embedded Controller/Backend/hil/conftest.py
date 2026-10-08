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
    """The rig, for a test that must arm it."""

    if not ALLOW_ARM:
        pytest.skip("arms the board: set TESTBED_HIL_ALLOW_ARM=1 once the rig is safe to energise")

    if safe_rig.mode() == "FAULT":
        pytest.fail(f"the board is in FAULT before the test: {safe_rig.faults()}")

    return safe_rig
