"""The serial link's framing, compiled from include/line_protocol.h and checked against the backend.

Both ends must compute the same CRC for every line, or every command would be refused and
every reading dropped. The harness frames lines the way the firmware does and the test
compares each one with the backend's frame_line(), then checks the firmware's verify()
against good, corrupted and unframed lines.
"""

import random
import shutil
import subprocess
from pathlib import Path

import pytest

INCLUDE_DIR = Path(__file__).resolve().parents[2] / "include"

HARNESS = r"""
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include "line_protocol.h"

using namespace line_protocol;

// Reads lines from stdin and prints each one framed the way the firmware's send_line() does.
// With "verify" as the argument, prints each line's verdict and body length instead.
int main(int argc, char** argv) {
  const bool verifying = argc > 1 && std::strcmp(argv[1], "verify") == 0;
  std::string line;

  while (std::getline(std::cin, line)) {
    if (verifying) {
      size_t body = 0;
      const Check check = verify(line.c_str(), line.size(), &body);
      std::printf("%s %u\n", check == Check::Valid ? "valid" : check == Check::Invalid ? "invalid" : "absent", (unsigned)body);
    } else {
      char suffix[CHECKSUM_LENGTH + 1];
      format_suffix(crc16(line.c_str(), line.size()), suffix);
      std::printf("%s%s\n", line.c_str(), suffix);
    }
  }

  return 0;
}
"""


@pytest.fixture(scope="module")
def harness(tmp_path_factory):
    if shutil.which("g++") is None:
        pytest.skip("needs a host C++ compiler")

    work = tmp_path_factory.mktemp("line_protocol")
    source, binary = work / "line_protocol.cpp", work / "line_protocol"
    source.write_text(HARNESS)

    build = subprocess.run(["g++", "-std=gnu++11", "-Wall", "-Wextra", "-Werror", f"-I{INCLUDE_DIR}", "-o", str(binary), str(source)],
                           capture_output=True, text=True)
    assert build.returncode == 0, build.stderr

    def run(lines, *args):
        result = subprocess.run([str(binary), *args], input="\n".join(lines) + "\n", capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stderr
        return result.stdout.splitlines()

    return run


def sample_lines():
    rng = random.Random(7)
    fixed = ["123456789", "PING", "ARM", "TARGETS 1.000000 0.500000 0.000000 3.300000 2.000000", "SWITCH_PERIOD_US 1",
             "MEASURED 1.23456 V seq=17 t_ms=850", "FAULTS mode=SAFE active=none latched=none history=none", ""]
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 =._-:,!*"
    randoms = ["".join(rng.choice(alphabet) for _ in range(rng.randint(1, 120))) for _ in range(300)]
    return fixed + randoms


def test_the_crc_is_the_standard_ccitt_false(harness):
    assert harness(["123456789"]) == ["123456789*29B1"]


def test_firmware_and_backend_frame_every_line_identically(backend_module, harness):
    lines = sample_lines()
    assert harness(lines) == [backend_module.frame_line(line) for line in lines]


def test_the_firmware_accepts_good_lines_and_rejects_corrupted_ones(backend_module, harness):
    framed = backend_module.frame_line("TARGETS 1.000000 0.500000 0.000000 3.300000 2.000000")
    corrupted = framed.replace("3.300000", "8.300000")
    lowercase = framed[:-4] + framed[-4:].lower()

    verdicts = harness([framed, corrupted, lowercase, "PING", "TOTAL*GARBAGE", "x*12", "*FFFF", "*0000"], "verify")

    assert verdicts == [
        f"valid {len(framed) - 5}",
        f"invalid {len(corrupted) - 5}",
        f"valid {len(lowercase) - 5}",
        "absent 4",          # typed at a terminal: no CRC, whole line is the command
        "absent 13",         # a '*' not followed by four hex digits is just text
        "absent 4",          # too short to carry one
        "valid 0",           # an empty line framed correctly (the CRC of nothing is FFFF); the firmware ignores it
        "invalid 0",
    ]


def test_every_single_bit_flip_in_a_command_is_caught(backend_module, harness):
    framed = backend_module.frame_line("SWITCH_PERIOD_US 5")
    body = framed[:-5].encode()

    flipped = []
    for index in range(len(body)):
        for bit in range(7):  # ASCII only, as on the wire
            corrupted = bytearray(body)
            corrupted[index] ^= 1 << bit
            text = corrupted.decode("ascii", "replace")
            if "\n" not in text and "\r" not in text:
                flipped.append(text + framed[-5:])

    assert set(line.split()[0] for line in harness(flipped, "verify")) == {"invalid"}
