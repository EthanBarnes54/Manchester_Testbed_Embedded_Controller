"""Property tests of the backend's handling of whatever arrives from the serial port.

The reader thread hands every line to _accept_line, _handle_board_line and _match_reply.
None of them may raise on any input, and nothing the CRC rejects may change what the
backend believes about the board."""

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

PREFIXES = ["MEASURED", "PINS", "ACK", "ACK ARM", "ACK DISARM", "ACK PIN", "ACK CLEAR FAULTS", "ERROR", "ERROR: Cannot arm",
            "ERROR: Not armed", "OK", "FAULT", "FAULTS", "FAILSAFE outputs zeroed", "HEALTH", "SELFTEST", "VERSION"]

TOKEN = st.one_of(
    st.text(alphabet=st.characters(min_codepoint=33, max_codepoint=126), max_size=12),
    st.builds(lambda key, value: f"{key}={value}",
              st.sampled_from(["mode", "seq", "t_ms", "protocol", "firmware", "active", "squeeze_plate", "pin_1", ""]),
              st.one_of(st.integers(-10**12, 10**12).map(str), st.floats(allow_nan=True).map(str),
                        st.sampled_from(["ARMED", "SAFE", "FAULT", "", "=", "nan", "inf", "-1"]))),
    st.floats(allow_nan=True, allow_infinity=True).map(str),
)

# Lines shaped like the board's, with arbitrary fields, plus arbitrary text. Readings get
# their own generator so the voltage field is often something float() accepts but is not
# a reading.
BOARD_LIKE = st.builds(lambda prefix, tokens: " ".join([prefix, *tokens]), st.sampled_from(PREFIXES), st.lists(TOKEN, max_size=8))
READING = st.builds(lambda volts, rest: " ".join(["MEASURED", volts, "V", *rest]),
                    st.one_of(st.floats(allow_nan=True, allow_infinity=True).map(str), st.sampled_from(["1e400", "-1e400", "NaN", "Infinity"]), TOKEN),
                    st.lists(TOKEN, max_size=3))
ANY_LINE = st.one_of(BOARD_LIKE, READING, st.text(max_size=120))


@pytest.fixture
def backend(backend_module):
    backend = backend_module.SerialBackend(port="FUZZ0", status=False)
    backend.online_update_enabled = False
    return backend


def feed(backend, line, timestamp=1000.0):
    """What the reader thread does with one received line."""

    message = backend._accept_line(line)

    if message is not None:
        backend._handle_board_line(timestamp, message)
        backend._match_reply(message)

    return message


@pytest.mark.req("LINK-01", "DATA-01")
def test_no_line_from_the_board_breaks_the_parsers_or_stores_a_non_reading(backend_module, backend, monkeypatch):
    stored = []
    append = backend._append_measurement
    monkeypatch.setattr(backend, "_append_measurement", lambda timestamp, voltage, message, **kw: (
        stored.append(voltage), append(timestamp, voltage, message, **kw)))

    @settings(max_examples=400, deadline=None)
    @given(lines=st.lists(st.one_of(ANY_LINE, ANY_LINE.map(backend_module.frame_line)), max_size=12))
    def check(lines):
        stored.clear()

        for line in lines:
            feed(backend, line)

        assert isinstance(backend.board_mode, str)
        assert all(voltage is None or math.isfinite(voltage) for voltage in stored), stored

    check()


@pytest.mark.req("DATA-01")
@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "1e400"])
def test_a_reading_that_is_not_a_number_is_stored_as_missing(backend, value):
    feed(backend, f"MEASURED {value} V seq=1 t_ms=50")
    data = backend.get_data()
    assert len(data) == 1 and data["voltage"].isna().all()


@pytest.mark.req("LINK-01", "SAF-02")
def test_a_line_the_crc_rejects_never_changes_what_the_backend_believes(backend_module, backend):
    backend.protocol_ok = True

    @settings(max_examples=300, deadline=None)
    @given(line=BOARD_LIKE, data=st.data())
    def check(line, data):
        backend.board_mode = "SAFE"
        backend.board_faults = {}
        framed = backend_module.frame_line(line)
        index = data.draw(st.integers(0, len(line) - 1))
        replacement = data.draw(st.sampled_from("#~Q").filter(lambda c: c != line[index]))
        corrupted = line[:index] + replacement + line[index + 1:] + framed[-5:]

        before = dict(backend.link_stats)
        assert feed(backend, corrupted) is None
        assert backend.board_mode == "SAFE" and backend.board_faults == {}
        assert backend.link_stats["bad_checksums"] == before["bad_checksums"] + 1

    check()


@pytest.mark.req("LINK-01", "SAF-02")
def test_once_the_protocol_is_confirmed_an_unframed_protocol_line_is_ignored(backend):
    backend.protocol_ok = True
    backend.board_mode = "SAFE"

    for line in ("ACK ARM", "FAULTS mode=ARMED", "HEALTH mode=ARMED", "FAULT X CRITICAL mode=ARMED"):
        assert feed(backend, line) is None

    assert backend.board_mode == "SAFE"
