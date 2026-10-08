import configparser
import importlib.util
import os
import re
import struct
import subprocess
import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
FIRMWARE_DIR = BACKEND_DIR.parent
REPO_DIR = FIRMWARE_DIR.parent
CI_WORKFLOW = (REPO_DIR / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
PLATFORMIO_INI = (FIRMWARE_DIR / "platformio.ini").read_text(encoding="utf-8")

# Where PlatformIO keeps the Arduino core's build script, which adds the -Wno-error flags.
ARDUINO_BUILD_SCRIPT = (Path(os.environ.get("PLATFORMIO_CORE_DIR", Path.home() / ".platformio"))
                        / "packages" / "framework-arduinoespressif32" / "tools" / "platformio-build-esp32.py")

# What that script exempted when this was written (Arduino core 2.0.17).
KNOWN_EXEMPTIONS = {"deprecated-declarations", "unused-but-set-variable", "unused-function", "unused-variable"}


def load_tool(name):
    spec = importlib.util.spec_from_file_location(name, FIRMWARE_DIR / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_src_flags():
    parser = configparser.ConfigParser(interpolation=None)
    parser.read_string(PLATFORMIO_INI)
    return parser["env"]["build_src_flags"].split()


@pytest.mark.req("QA-01")
def test_project_sources_build_with_every_warning_as_an_error():
    flags = build_src_flags()
    assert {"-Wall", "-Wextra", "-Werror"} <= set(flags)

    exempted = KNOWN_EXEMPTIONS
    if ARDUINO_BUILD_SCRIPT.exists():
        exempted = set(re.findall(r'"-Wno-error=([\w-]+)"', ARDUINO_BUILD_SCRIPT.read_text(encoding="utf-8")))
        assert exempted, "the Arduino build script no longer exempts anything; check this test still makes sense"

    assert {f"-Werror={name}" for name in exempted} <= set(flags)


@pytest.mark.req("QA-01")
def test_ci_builds_every_environment():
    environments = set(re.findall(r"^\[env:(\w+)\]", PLATFORMIO_INI, re.M))
    built = re.search(r"pio run((?: -e \w+)+)", CI_WORKFLOW).group(1).split()[1::2]
    assert set(built) == environments


@pytest.mark.req("QA-02")
@pytest.mark.skipif(importlib.util.find_spec("ruff") is None, reason="ruff is not installed (requirements-dev.txt)")
def test_the_backend_passes_its_static_checks():
    result = subprocess.run([sys.executable, "-m", "ruff", "check", "--no-cache", "."], cwd=FIRMWARE_DIR,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.req("QA-02")
def test_the_coverage_floor_is_set_and_enforced_in_ci():
    config = configparser.ConfigParser()
    config.read(BACKEND_DIR / ".coveragerc")
    assert config.getfloat("report", "fail_under") >= 60
    assert "ruff check" in CI_WORKFLOW
    assert re.search(r"pytest .*--cov\b", CI_WORKFLOW)


def partition_table(*app_sizes, data_size=0x5000):
    entries = [b"\xaa\x50" + bytes([0x01, 0x02]) + struct.pack("<II", 0x9000, data_size) + b"nvs".ljust(16, b"\0") + bytes(4)]
    entries += [b"\xaa\x50" + bytes([0x00, 0x10 + index]) + struct.pack("<II", 0x10000, size) + f"app{index}".encode().ljust(16, b"\0") + bytes(4)
                for index, size in enumerate(app_sizes)]
    return b"".join(entries) + b"\xff" * 32


@pytest.mark.req("QA-03")
@pytest.mark.parametrize("image_bytes, within", [(800_000, True), (838_861, False), (1_400_000, False)])
def test_an_image_over_its_share_of_the_app_partition_fails(tmp_path, image_bytes, within):
    tool = load_tool("check_firmware_size")
    build = tmp_path / "esp32dev"
    build.mkdir()
    (build / "firmware.bin").write_bytes(bytes(image_bytes))
    # Two app slots of different sizes: the image must fit the smaller one.
    (build / "partitions.bin").write_bytes(partition_table(1_048_576, 1_310_720))

    ok, report = tool.check(build, budget=0.80)
    assert ok is within, report
    assert "1048576" in report


@pytest.mark.req("QA-03")
def test_a_missing_build_fails_the_size_check(tmp_path):
    tool = load_tool("check_firmware_size")
    (tmp_path / "esp32dev").mkdir()
    assert tool.main(["esp32dev", "--build-dir", str(tmp_path)]) == 1


@pytest.mark.req("QA-03")
def test_ci_checks_the_size_of_every_image():
    environments = set(re.findall(r"^\[env:(\w+)\]", PLATFORMIO_INI, re.M))
    checked = re.search(r"check_firmware_size\.py((?: \w+)+)", CI_WORKFLOW).group(1).split()
    assert set(checked) == environments


@pytest.mark.req("QA-04")
def test_ci_runs_the_mutation_suite():
    assert re.search(r"pytest .*-m mutation", CI_WORKFLOW)


@pytest.mark.req("QA-03")
def test_the_size_check_reads_a_real_partition_table():
    """When a firmware build is present locally, the tool parses its real partition table."""

    built = sorted((FIRMWARE_DIR / ".pio" / "build").glob("*/partitions.bin"))
    if not built:
        pytest.skip("no firmware has been built here")

    sizes = load_tool("check_firmware_size").app_partition_sizes(built[0].read_bytes())
    assert sizes and all(size >= 0x100000 for size in sizes)
