import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "run_relabel_api", Path(__file__).resolve().parents[1] / "scripts" / "run_relabel_api.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_parse_lines_tolerates_fences_and_reorders():
    text = '```\n{"id": "b", "label": 0, "note": "chê"}\nrác\n{"id": "a", "label": 2, "note": "khen"}\n```'
    assert mod.parse_lines(text, ["a", "b"]) == [
        {"id": "a", "label": 2, "note": "khen"}, {"id": "b", "label": 0, "note": "chê"}]


def test_parse_lines_rejects_missing_or_invalid():
    assert mod.parse_lines('{"id": "a", "label": 2, "note": "x"}', ["a", "b"]) is None
    assert mod.parse_lines('{"id": "a", "label": 5, "note": "x"}', ["a"]) is None
    assert mod.parse_lines('{"id": "a", "label": 1, "note": ""}', ["a"]) is None
