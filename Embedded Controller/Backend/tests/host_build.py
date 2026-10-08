"""Builds and runs the host harnesses that test firmware code on the PC.

Every harness goes through here, so they all get the same strict warnings and, where the
toolchain has them, the same sanitizers: AddressSanitizer and UndefinedBehaviorSanitizer
with their runtime where it exists (Linux, so CI), or UndefinedBehaviorSanitizer in trap
mode, which needs no runtime (MinGW). A trap shows up as the harness crashing rather than
as a report. TESTBED_HOST_SANITIZE=0 turns them off, to tell a sanitizer finding apart
from an ordinary failed check.
"""

import functools
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
FIRMWARE_DIR = BACKEND_DIR.parent
INCLUDE_DIR = FIRMWARE_DIR / "include"
MAIN_CPP_PATH = FIRMWARE_DIR / "src" / "main.cpp"

COMPILER = "g++"
WARNINGS = ("-Wall", "-Wextra", "-Werror")

# Tried in order; the first that builds and runs a clean program is used.
SANITIZER_CHOICES = (
    ("-fsanitize=address,undefined", "-fno-sanitize-recover=all", "-fno-omit-frame-pointer"),
    ("-fsanitize=undefined", "-fsanitize-trap=all"),
)

needs_compiler = pytest.mark.skipif(shutil.which(COMPILER) is None, reason="needs a host C++ compiler")

_PROBE = "#include <cstdio>\nint main(int argc, char**) { std::printf(\"%d\\n\", argc + 1); return 0; }\n"


def main_cpp() -> str:
    return MAIN_CPP_PATH.read_text(encoding="utf-8")


@functools.lru_cache(maxsize=None)
def sanitizer_flags() -> tuple:
    """The sanitizer flags this toolchain supports, or () if none or turned off."""

    if os.environ.get("TESTBED_HOST_SANITIZE", "1") == "0" or shutil.which(COMPILER) is None:
        return ()

    with tempfile.TemporaryDirectory() as work:
        source, binary = Path(work) / "probe.cpp", Path(work) / "probe"
        source.write_text(_PROBE)

        for flags in SANITIZER_CHOICES:
            built = subprocess.run([COMPILER, *flags, "-o", str(binary), str(source)], capture_output=True, text=True)

            if built.returncode == 0 and subprocess.run([str(binary)], capture_output=True).returncode == 0:
                return flags

    return ()


def compile_harness(source_text: str, work_dir: Path, name: str, *, std: str = "gnu++11", defines: dict | None = None,
                    include_dirs: tuple = (), optimise: bool = False,
                    warnings_as_errors: bool = True) -> tuple[subprocess.CompletedProcess, Path]:
    """Compiles one harness. include_dirs are searched before include/, so a mutated copy of
    a header there shadows the real one. warnings_as_errors=False is for the mutation tests,
    whose deliberately broken code often draws a warning but must still be judged by whether
    the checks notice. Returns the compiler's result and the binary path."""

    source, binary = work_dir / f"{name}.cpp", work_dir / name
    source.write_text(source_text, encoding="utf-8")

    warnings = WARNINGS if warnings_as_errors else tuple(flag for flag in WARNINGS if flag != "-Werror")
    command = [COMPILER, f"-std={std}", *warnings, *sanitizer_flags()]
    command += ["-O2"] if optimise else []
    command += [f"-D{key}={value}" for key, value in (defines or {}).items()]
    command += [f"-I{directory}" for directory in (*include_dirs, INCLUDE_DIR)]
    command += ["-o", str(binary), str(source)]

    return subprocess.run(command, capture_output=True, text=True), binary


def build(source_text: str, work_dir: Path, name: str, **options) -> Path:
    """compile_harness, failing the test with the compiler's output if it does not build."""

    built, binary = compile_harness(source_text, work_dir, name, **options)
    assert built.returncode == 0, built.stderr
    return binary


def run(binary: Path, *args, input: str | None = None, timeout: float = 120) -> subprocess.CompletedProcess:
    return subprocess.run([str(binary), *map(str, args)], input=input, capture_output=True, text=True, timeout=timeout)


def build_and_run(source_text: str, work_dir: Path, name: str, **options) -> subprocess.CompletedProcess:
    """Builds a self-checking harness and runs it once. Its exit code is the verdict."""

    return run(build(source_text, work_dir, name, **options))
