import importlib.util
from pathlib import Path

FIRMWARE_DIR = Path(__file__).resolve().parents[2]


def load_traceability():
    spec = importlib.util.spec_from_file_location("traceability", FIRMWARE_DIR / "tools" / "traceability.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_requirement_is_traced_and_every_marker_names_a_requirement():
    _, problems = load_traceability().trace()
    assert problems == []


def test_the_check_finds_an_untested_requirement_an_unknown_id_and_a_bench_step_with_no_procedure(tmp_path):
    traceability = load_traceability()
    requirements_md = tmp_path / "requirements.md"
    requirements_md.write_text(
        "| ID | Requirement | Verification |\n|----|----|----|\n"
        "| SW-01 | Tested and claimed. | Test |\n"
        "| SW-02 | Tested but never claimed. | Test |\n"
        "| SW-03 | At the bench, with no step named. | Bench |\n"
        "| SW-04 | At a step the procedure lacks. | Bench ATP-99 |\n"
        "| SW-05 | By reading the design. | Inspection |\n", encoding="utf-8")

    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_example.py").write_text(
        "import pytest\n\n"
        "@pytest.mark.req(\"SW-01\")\ndef test_claimed():\n    pass\n\n"
        "@pytest.mark.req(\"SW-77\")\ndef test_unknown():\n    pass\n", encoding="utf-8")

    _, problems = traceability.trace(traceability.read_requirements(requirements_md), traceability.read_markers([tests]),
                                     procedure_steps={"ATP-01"})

    assert len(problems) == 4
    assert any("SW-02" in problem and "no test" in problem for problem in problems)
    assert any("SW-77" in problem for problem in problems)
    assert any("SW-03" in problem and "names no ATP step" in problem for problem in problems)
    assert any("ATP-99" in problem for problem in problems)


def test_a_rig_test_alone_does_not_verify_a_test_requirement(tmp_path):
    traceability = load_traceability()
    requirements_md = tmp_path / "requirements.md"
    requirements_md.write_text("| SAF-03 | Fails safe. | Test; Bench ATP-04 |\n", encoding="utf-8")
    hil = tmp_path / "hil"
    hil.mkdir()
    (hil / "test_rig.py").write_text("import pytest\n\n@pytest.mark.req(\"SAF-03\")\ndef test_on_the_rig():\n    pass\n",
                                     encoding="utf-8")

    _, problems = traceability.trace(traceability.read_requirements(requirements_md), traceability.read_markers([hil]),
                                     procedure_steps={"ATP-04"})
    assert problems == ['SAF-03 is verified by Test, but no test in tests/ carries @pytest.mark.req("SAF-03")']


def test_a_module_level_marker_covers_every_test_in_the_module(tmp_path):
    traceability = load_traceability()
    (tmp_path / "test_module.py").write_text(
        "import pytest\n\npytestmark = [pytest.mark.hil, pytest.mark.req(\"SAF-03\")]\n\n"
        "def test_one():\n    pass\n\ndef test_two():\n    pass\n", encoding="utf-8")

    markers = traceability.read_markers([tmp_path])
    assert sorted(markers.values()) == [["SAF-03"], ["SAF-03"]]
