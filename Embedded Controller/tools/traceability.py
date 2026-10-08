"""Requirements traceability: which tests verify which requirement, and what is missing.

Reads the requirement tables in docs/requirements.md and the @pytest.mark.req(...) markers
on the tests (statically, without importing or running them), then reports:

- every requirement whose verification includes Test but which no test claims;
- every marker naming an ID that is not a requirement;
- every Bench requirement that names no acceptance test step;
- every acceptance test step named that docs/acceptance-test-procedure.md does not define,
  once that document exists.

    python tools/traceability.py            prints the matrix as Markdown
    python tools/traceability.py --check    prints only the problems; exits 1 if there are any
"""

import ast
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

FIRMWARE_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = FIRMWARE_DIR.parent
REQUIREMENTS = REPO_DIR / "docs" / "requirements.md"
PROCEDURE = REPO_DIR / "docs" / "acceptance-test-procedure.md"
TEST_DIRS = (FIRMWARE_DIR / "Backend" / "tests", FIRMWARE_DIR / "Backend" / "hil")

METHODS = ("Test", "Bench", "Inspection", "Analysis")
ROW = re.compile(r"^\|\s*([A-Z]+-\d{2})\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*$")
ATP = re.compile(r"\bATP-\d{2}\b")


@dataclass
class Requirement:
    id: str
    text: str
    methods: tuple
    procedures: tuple
    tests: list = field(default_factory=list)


def read_requirements(path: Path = REQUIREMENTS) -> dict:
    requirements = {}

    for line in path.read_text(encoding="utf-8").splitlines():
        match = ROW.match(line)

        if not match:
            continue

        identifier, text, verification = match.groups()

        if identifier in requirements:
            raise ValueError(f"{identifier} is defined twice in {path.name}")

        methods = tuple(method for method in METHODS if re.search(rf"\b{method}\b", verification))

        if not methods:
            raise ValueError(f"{identifier} names no verification method: {verification!r}")

        requirements[identifier] = Requirement(identifier, text, methods, tuple(ATP.findall(verification)))

    return requirements


def _req_ids(decorator) -> list:
    """The IDs in a pytest.mark.req(...) expression, or [] if it is something else."""

    if not isinstance(decorator, ast.Call):
        return []

    target = decorator.func

    if not (isinstance(target, ast.Attribute) and target.attr == "req"):
        return []

    return [argument.value for argument in decorator.args
            if isinstance(argument, ast.Constant) and isinstance(argument.value, str)]


def read_markers(test_dirs=TEST_DIRS) -> dict:
    """{test id: [requirement IDs]} for every test function carrying a req marker. A module
    level pytestmark applies its IDs to every test in that module."""

    markers = {}

    for directory in test_dirs:
        for path in sorted(Path(directory).glob("test_*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
            module_ids = []

            for node in tree.body:
                if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "pytestmark" for t in node.targets):
                    values = node.value.elts if isinstance(node.value, (ast.List, ast.Tuple)) else [node.value]
                    for value in values:
                        module_ids += _req_ids(value)

            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
                    ids = module_ids + [i for decorator in node.decorator_list for i in _req_ids(decorator)]

                    if ids:
                        markers[f"{path.parent.name}/{path.name}::{node.name}"] = ids

    return markers


def read_procedure_steps(path: Path = PROCEDURE) -> set | None:
    """The ATP-NN steps the acceptance test procedure defines (as headings), or None if it
    does not exist yet."""

    if not path.exists():
        return None

    return set(re.findall(r"^#+\s*(ATP-\d{2})\b", path.read_text(encoding="utf-8"), re.M))


def trace(requirements=None, markers=None, procedure_steps=...) -> tuple[dict, list]:
    """Attaches tests to requirements; returns (requirements, problems)."""

    requirements = read_requirements() if requirements is None else requirements
    markers = read_markers() if markers is None else markers
    procedure_steps = read_procedure_steps() if procedure_steps is ... else procedure_steps
    problems = []

    for test, ids in sorted(markers.items()):
        for identifier in ids:
            if identifier in requirements:
                requirements[identifier].tests.append(test)
            else:
                problems.append(f"{test} names {identifier}, which is not in requirements.md")

    for requirement in requirements.values():
        # The rig tests in hil/ never run in CI, so they count towards Bench, not Test.
        if "Test" in requirement.methods and not any(not test.startswith("hil/") for test in requirement.tests):
            problems.append(f"{requirement.id} is verified by Test, but no test in tests/ carries "
                            f"@pytest.mark.req(\"{requirement.id}\")")

        if "Bench" in requirement.methods and not requirement.procedures:
            problems.append(f"{requirement.id} is verified at the Bench but names no ATP step")

        if procedure_steps is not None:
            for step in requirement.procedures:
                if step not in procedure_steps:
                    problems.append(f"{requirement.id} names {step}, which acceptance-test-procedure.md does not define")

    return requirements, problems


def matrix(requirements: dict) -> str:
    rows = ["| Requirement | Verification | Bench steps | Tests |", "|---|---|---|---|"]

    for requirement in requirements.values():
        tests = "<br>".join(f"`{test}`" for test in requirement.tests) or "—"
        rows.append(f"| {requirement.id} | {', '.join(requirement.methods)} | {', '.join(requirement.procedures) or '—'} | {tests} |")

    return "\n".join(rows)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    requirements, problems = trace()

    if "--check" not in argv:
        print("# Traceability matrix\n")
        print(f"Generated from docs/requirements.md and the tests' req markers. {len(requirements)} requirements.\n")
        print(matrix(requirements))
        print()

    for problem in problems:
        print(f"PROBLEM: {problem}", file=sys.stderr if "--check" not in argv else sys.stdout)

    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
