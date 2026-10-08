import subprocess

Import("projenv")  # noqa: F821 - provided by PlatformIO


def git_describe():
    try:
        return subprocess.check_output(
            ["git", "describe", "--always", "--dirty", "--tags"],
            cwd=projenv.subst("$PROJECT_DIR"),  # noqa: F821
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


projenv.Append(  # noqa: F821
    CPPDEFINES=[
        ("TESTBED_FIRMWARE_VERSION", projenv.StringifyMacro(git_describe())),  # noqa: F821
        ("TESTBED_BUILD_ENV", projenv.StringifyMacro(projenv["PIOENV"])),  # noqa: F821
    ]
)
