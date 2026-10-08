from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
from urllib.parse import quote
import uuid

PROJECT_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_DIR / "Backend"


def python_requirements(*files):
    """(name, version) for every exact pin in the given requirements files, following -r includes."""

    pins, seen = {}, set()

    def read(path):
        if path in seen:
            return
        seen.add(path)

        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()

            if line.startswith("-r "):
                read(path.parent / line[3:].strip())
            elif "==" in line:
                name, version = (part.strip() for part in line.split("==", 1))
                pins[name.lower()] = (name, version)

    for path in files:
        read(Path(path))

    return sorted(pins.values(), key=lambda pin: pin[0].lower())


def platformio_pins(ini_text):
    """(platform, version) and [(owner/name, version)] as platformio.ini pins them."""

    platform = re.search(r"^platform\s*=\s*([\w-]+)@(\S+)", ini_text, re.M)
    libraries = re.findall(r"^\s+([\w-]+/[^@\n]+)@(\S+)\s*$", ini_text, re.M)
    return (platform.group(1), platform.group(2)) if platform else None, [(name.strip(), version) for name, version in libraries]


def installed_framework():
    """The Arduino core PlatformIO installed, if it is here to read."""

    manifest = Path.home() / ".platformio" / "packages" / "framework-arduinoespressif32" / "package.json"

    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
        return data["name"], data["version"]
    except (OSError, ValueError, KeyError):
        return None


def component(kind, name, version, purl_type):
    # Package URLs percent-encode anything outside the name's own path separators.
    purl = f"pkg:{purl_type}/{quote(name, safe='/')}@{quote(version, safe='')}"
    return {"type": kind, "bom-ref": purl, "name": name, "version": version, "purl": purl}


def build_sbom():
    components = [
        component("library", name.lower(), version, "pypi")
        for name, version in python_requirements(BACKEND_DIR / "requirements.txt", BACKEND_DIR / "requirements-dev.txt")
    ]

    platform, libraries = platformio_pins((PROJECT_DIR / "platformio.ini").read_text(encoding="utf-8"))

    if platform:
        components.append(component("platform", platform[0], platform[1], "platformio"))

    for name, version in libraries:
        components.append(component("library", name, version, "platformio"))

    framework = installed_framework()

    if framework:
        components.append(component("framework", framework[0], framework[1], "platformio"))

    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "serialNumber": f"urn:uuid:{uuid.uuid4()}",
        "version": 1,
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "component": {"type": "application", "bom-ref": "manchester-testbed-controller", "name": "Manchester Testbed Embedded Controller"},
        },
        "components": components,
    }


def main(argv):
    output = Path(argv[1]) if len(argv) > 1 else PROJECT_DIR / "sbom.cdx.json"
    sbom = build_sbom()
    output.write_text(json.dumps(sbom, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(sbom['components'])} components to {output}")


if __name__ == "__main__":
    main(sys.argv)
