"""A minimal client for the board's serial protocol, for the hardware-in-the-loop tests.

Deliberately independent of python_Backend: it frames and checks lines from the protocol
definition (CRC-16/CCITT-FALSE, "*XXXX"), so the tests check the board against the
protocol rather than against the backend's reading of it.
"""

import binascii
import contextlib
import re
import threading
import time

import serial

BAUD_RATE = 115200
PROTOCOL_PREFIXES = ("MEASURED", "PINS", "ACK", "ERROR", "OK", "FAULT", "FAILSAFE", "HEALTH", "SELFTEST", "VERSION")
UNSOLICITED_ERRORS = ("ERROR: ADC conversion timed out", "ERROR: OTA")
CHECKSUM = re.compile(r"\*[0-9A-Fa-f]{4}$")


def frame(text: str) -> str:
    return f"{text}*{binascii.crc_hqx(text.encode('utf-8'), 0xFFFF):04X}"


def unframe(line: str) -> tuple[str, str]:
    """(text, "valid" | "invalid" | "absent")"""

    if len(line) >= 5 and CHECKSUM.search(line):
        text = line[:-5]
        return text, ("valid" if int(line[-4:], 16) == binascii.crc_hqx(text.encode("utf-8"), 0xFFFF) else "invalid")

    return line, "absent"


def fields(line: str) -> dict:
    """The key=value fields of a board line."""

    return dict(token.split("=", 1) for token in line.split() if "=" in token)


class Received:
    __slots__ = ("time", "text", "verdict")

    def __init__(self, at, text, verdict):
        self.time, self.text, self.verdict = at, text, verdict


class Rig:
    """One open connection to the board. Every line it sends is recorded with the time it
    was sent; every line received is kept with the time it arrived and its CRC verdict."""

    def __init__(self, port: str, reset: bool = True):
        if port == "sim":
            # The protocol-level simulator, for checking these tests without a board.
            from simulated_board import SimulatedBoard
            self.serial = SimulatedBoard()
        else:
            self.serial = serial.Serial()

        self.serial.port, self.serial.baudrate, self.serial.timeout = port, BAUD_RATE, 0.05
        # Held released before opening, so opening the port does not itself reset the board.
        self.serial.dtr = False
        self.serial.rts = False
        self.serial.open()

        self.received: list[Received] = []
        self.last_sent_at = 0.0
        self._condition = threading.Condition()
        self._write_lock = threading.Lock()
        self._closing = threading.Event()
        self._reader = threading.Thread(target=self._read_forever, name="hil-reader", daemon=True)
        self._reader.start()

        if reset:
            self.reset()
        else:
            self.wait_until_ready()

    # ---------------------------------------------------------------- link

    def _read_forever(self):
        while not self._closing.is_set():
            try:
                raw = self.serial.readline()
            except serial.SerialException:
                return

            if not raw:
                continue

            text, verdict = unframe(raw.decode("utf-8", "replace").strip())

            with self._condition:
                self.received.append(Received(time.monotonic(), text, verdict))
                self._condition.notify_all()

    def send(self, text: str, framed: bool = True):
        self.send_raw(((frame(text) if framed else text) + "\n").encode("utf-8"))

    def send_raw(self, data: bytes):
        with self._write_lock:
            self.serial.write(data)
            self.serial.flush()
            self.last_sent_at = time.monotonic()

    def mark(self) -> int:
        """A position in the received lines, so a wait only looks at what came after it."""

        with self._condition:
            return len(self.received)

    def wait_for(self, predicate, since: int, timeout: float) -> Received | None:
        """The first line received after since that satisfies predicate, or None on timeout."""

        deadline = time.monotonic() + timeout

        with self._condition:
            index = since

            while True:
                while index < len(self.received):
                    line = self.received[index]
                    index += 1

                    if predicate(line):
                        return line

                remaining = deadline - time.monotonic()

                if remaining <= 0:
                    return None

                self._condition.wait(remaining)

    def lines(self, since: int = 0, prefix: str = "") -> list[Received]:
        with self._condition:
            return [line for line in self.received[since:] if line.text.startswith(prefix)]

    def command(self, text: str, replies: tuple, timeout: float = 1.0, framed: bool = True) -> str:
        """Sends a command and returns the first reply starting with one of replies, or an
        ERROR that answers it. Fails if nothing answers within the timeout."""

        since = self.mark()
        self.send(text, framed=framed)

        def answers(line):
            return line.verdict != "invalid" and (line.text.startswith(replies) or (
                line.text.startswith("ERROR") and not line.text.startswith(UNSOLICITED_ERRORS)))

        reply = self.wait_for(answers, since, timeout)
        assert reply is not None, f"the board did not answer {text!r} within {timeout} s"
        return reply.text

    # ----------------------------------------------------------- board state

    def version(self) -> dict:
        return fields(self.command("VERSION", ("VERSION ",)))

    def faults(self) -> dict:
        return fields(self.command("FAULTS", ("FAULTS ",)))

    def health(self) -> dict:
        return fields(self.command("HEALTH", ("HEALTH ",)))

    def selftest(self) -> str:
        return self.command("SELFTEST", ("SELFTEST ",), timeout=3.0)

    def mode(self) -> str:
        return self.faults()["mode"]

    def make_safe(self):
        """DISARM and zero every output; leaves the board SAFE (or FAULT if a fault is latched)."""

        self.command("SWITCH_PERIOD_US 0", ("ACK SWITCH_PERIOD_US",))
        self.command("TARGETS 0 0 0 0 0", ("ACK TARGETS",))
        self.command("DISARM", ("ACK DISARM",))

    # ------------------------------------------------------------- control

    def reset(self, timeout: float = 15.0):
        """Resets the board through the USB bridge's RTS line (wired to EN on ESP32 dev
        boards, as esptool uses it) and waits until it answers."""

        self.serial.dtr = False
        self.serial.rts = True
        time.sleep(0.1)
        self.serial.rts = False
        self.wait_until_ready(timeout)

    def wait_until_ready(self, timeout: float = 15.0):
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            since = self.mark()
            self.send("VERSION")

            if self.wait_for(lambda line: line.text.startswith("VERSION ") and line.verdict == "valid", since, 1.0):
                return

        raise AssertionError(f"the board did not answer VERSION within {timeout} s of a reset")

    @contextlib.contextmanager
    def keepalive(self, interval: float = 1.0):
        """Sends PING every interval, as the backend does, so the board's failsafe stays quiet."""

        stop = threading.Event()

        def ping():
            while not stop.wait(interval):
                self.send("PING")

        thread = threading.Thread(target=ping, name="hil-keepalive", daemon=True)
        thread.start()

        try:
            yield
        finally:
            stop.set()
            thread.join()

    def close(self):
        self._closing.set()
        self.serial.close()
        self._reader.join(timeout=2)
