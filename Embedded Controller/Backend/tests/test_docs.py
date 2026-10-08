import ast
import re
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
FIRMWARE_DIR = BACKEND_DIR.parent
DOCS = FIRMWARE_DIR.parent / "docs"

MAIN_CPP = (FIRMWARE_DIR / "src" / "main.cpp").read_text(encoding="utf-8")
SAFETY_STATE_H = (FIRMWARE_DIR / "include" / "safety_state.h").read_text(encoding="utf-8")
PROTOCOL = (DOCS / "protocol.md").read_text(encoding="utf-8")
REQUIREMENTS = (DOCS / "requirements.md").read_text(encoding="utf-8")
PROCEDURE = (DOCS / "acceptance-test-procedure.md").read_text(encoding="utf-8")
HAZARDS = (DOCS / "safety" / "hazard-analysis.md").read_text(encoding="utf-8")

REQUIREMENT_IDS = set(re.findall(r"^\|\s*([A-Z]+-\d{2})\s*\|", REQUIREMENTS, re.M))
# Anything in a document that looks like a requirement ID: one of the requirement prefixes.
REQUIREMENT_REFERENCE = re.compile(r"\b(?:%s)-\d{2}\b" % "|".join(sorted({i.split("-")[0] for i in REQUIREMENT_IDS})))
PROCEDURE_STEPS = set(re.findall(r"^## (ATP-\d{2})\b", PROCEDURE, re.M))


def firmware_commands():
    handler = MAIN_CPP[MAIN_CPP.index("void handle_command("):]
    handler = handler[:handler.index("\n  }\n")]
    return set(re.findall(r'command\.(?:equalsIgnoreCase|startsWith)\("([A-Z_ ]+)"\)', handler))


def test_the_protocol_document_lists_every_command_the_firmware_accepts():
    commands = firmware_commands()
    assert {"PING", "ARM", "TARGETS", "SWITCH_PERIOD_US"} <= commands
    table = PROTOCOL[PROTOCOL.index("## Commands"):PROTOCOL.index("## Lines the board sends")]
    documented = set()
    for row in re.findall(r"^\| (.*?) \| ", table, re.M):
        # A command is its upper-case words; its arguments start lower case or with '<'.
        documented |= set(re.findall(r"`([A-Z_]+(?: [A-Z_]+)*)(?: [^`]*)?`", row))
    assert commands - documented == set()


def test_the_protocol_document_names_the_firmware_protocol_version():
    firmware = re.search(r"constexpr int PROTOCOL_VERSION = (\d+);", MAIN_CPP).group(1)
    assert re.search(rf"Protocol version \*\*{firmware}\*\*", PROTOCOL)


def test_the_protocol_document_lists_every_fault_with_its_severity():
    names = set(re.findall(r'return "([A-Z_]+)";', SAFETY_STATE_H[SAFETY_STATE_H.index("name_of("):])) - {"UNKNOWN", "ARMED", "FAULT", "SAFE"}
    severities = dict(re.findall(r"^\| `([A-Z_]+)` \| (Critical|Warning) \|", PROTOCOL, re.M))
    assert set(severities) == names

    critical_block = SAFETY_STATE_H[SAFETY_STATE_H.index("severity_of("):SAFETY_STATE_H.index("return Severity::Critical;")]
    critical_enums = re.findall(r"case Fault::(\w+):", critical_block)
    critical_names = {re.sub(r"(?<!^)(?=[A-Z])", "_", name).upper() for name in critical_enums}
    assert {name for name, severity in severities.items() if severity == "Critical"} == critical_names


def test_every_requirement_and_bench_step_a_hazard_names_exists():
    named = set(REQUIREMENT_REFERENCE.findall(HAZARDS))
    assert named and named - REQUIREMENT_IDS == set()
    assert set(re.findall(r"\bATP-\d{2}\b", HAZARDS)) - PROCEDURE_STEPS == set()


def test_every_hazard_is_mitigated_by_at_least_one_requirement():
    rows = [line for line in HAZARDS.splitlines() if re.match(r"^\| H-\d{2} \|", line)]
    assert len(rows) >= 10
    for row in rows:
        assert REQUIREMENT_REFERENCE.search(row), row


def test_every_rig_test_the_procedure_names_exists():
    defined = set()
    for path in (BACKEND_DIR / "hil").glob("test_*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        defined |= {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
        defined.add(f"hil/{path.name}")

    named = set(re.findall(r"`(?:hil/test_rig\.py::)?(test_\w+)`", PROCEDURE)) | set(re.findall(r"`(hil/test_\w+\.py)`", PROCEDURE))
    assert named and named - defined == set()


def test_every_procedure_step_names_the_requirements_it_verifies():
    steps = re.split(r"^## (?=ATP-\d{2})", PROCEDURE, flags=re.M)[1:]
    assert len(steps) >= 13
    for step in steps:
        verifies = re.search(r"^Verifies (.+)$", step, re.M)
        assert verifies, step.splitlines()[0]
        ids = set(re.findall(r"[A-Z]+-\d{2}", verifies.group(1)))
        assert ids and ids <= REQUIREMENT_IDS, step.splitlines()[0]
        number = step.split()[0]
        for identifier in ids:
            row = re.search(rf"^\| {identifier} \|.*$", REQUIREMENTS, re.M).group(0)
            assert number in row, f"{identifier} does not name {number}"
        naming = set(re.findall(rf"^\| ([A-Z]+-\d{{2}}) \|.*\b{number}\b.*$", REQUIREMENTS, re.M))
        assert naming == ids, f"{number}: requirements naming it {sorted(naming)}, verified {sorted(ids)}"


def test_the_decision_record_index_lists_every_record():
    adr = DOCS / "adr"
    files = {path.name for path in adr.glob("[0-9][0-9][0-9][0-9]-*.md")}
    index = set(re.findall(r"\]\((\d{4}-[\w-]+\.md)\)", (adr / "README.md").read_text(encoding="utf-8")))
    assert files and files == index
