"""Fails a build whose firmware image leaves too little of its application partition free.

Over-the-air updates write the new image into the other application slot, so an image
that grows to fill its slot leaves no room for the fix that follows it. Reads each built
environment's firmware.bin and the partition table the build produced (partitions.bin),
and compares the image with the smallest application partition.

    python tools/check_firmware_size.py                     every environment under .pio/build
    python tools/check_firmware_size.py esp32dev --budget 0.8
"""

import argparse
import struct
import sys
from pathlib import Path

FIRMWARE_DIR = Path(__file__).resolve().parents[1]
BUILD_DIR = FIRMWARE_DIR / ".pio" / "build"

DEFAULT_BUDGET = 0.80

# ESP-IDF partition table: 32-byte entries starting with the magic AA 50; type 0 is an app.
ENTRY_SIZE = 32
ENTRY_MAGIC = b"\xaa\x50"
APP_TYPE = 0x00


def app_partition_sizes(table: bytes) -> list:
    sizes = []

    for start in range(0, len(table) - ENTRY_SIZE + 1, ENTRY_SIZE):
        entry = table[start:start + ENTRY_SIZE]

        if entry[:2] != ENTRY_MAGIC:
            break

        if entry[2] == APP_TYPE:
            sizes.append(struct.unpack_from("<I", entry, 8)[0])

    return sizes


def check(build_dir: Path, budget: float = DEFAULT_BUDGET) -> tuple[bool, str]:
    """(within budget, one-line report) for one environment's build directory."""

    image, table = build_dir / "firmware.bin", build_dir / "partitions.bin"

    if not image.exists() or not table.exists():
        return False, f"{build_dir.name}: no firmware.bin or partitions.bin; build it first"

    sizes = app_partition_sizes(table.read_bytes())

    if not sizes:
        return False, f"{build_dir.name}: the partition table has no application partition"

    used, slot = image.stat().st_size, min(sizes)
    share = used / slot
    verdict = "ok" if share <= budget else "OVER BUDGET"

    return share <= budget, f"{build_dir.name}: {used} of {slot} bytes ({share:.1%}, budget {budget:.0%}) {verdict}"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("environments", nargs="*", help="build environments to check (default: every one built)")
    parser.add_argument("--budget", type=float, default=DEFAULT_BUDGET, help="largest share of the partition an image may use")
    parser.add_argument("--build-dir", type=Path, default=BUILD_DIR)
    arguments = parser.parse_args(argv)

    environments = arguments.environments or sorted(path.name for path in arguments.build_dir.iterdir() if path.is_dir())
    results = [check(arguments.build_dir / environment, arguments.budget) for environment in environments]

    for _, report in results:
        print(report)

    return 0 if results and all(within for within, _ in results) else 1


if __name__ == "__main__":
    sys.exit(main())
