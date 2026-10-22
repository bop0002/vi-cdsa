import json

import pandas as pd
import pytest

from src.relabel import (apply_relabels, export_batches, merge_results, next_wave, qa_against_audit,
                         select_gray, validate_results)


def pool():
    return pd.DataFrame({
        "id": ["tiki_1", "tiki_2", "tiki_3", "res1_1", "res1_2"],
        "domain": ["A", "A", "A", "B", "B"],
        "rating": [3, 4, 5, -1, -1],
        "label": [1, 2, 2, 1, 2],
        "text_raw": ["Sách  rách\nbẩn quá", "Ổn nhưng giao chậm", "Tuyệt vời", "quán tạm được", "ngon lắm"],
    })


def write_results(results_dir, batch, rows):
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / batch).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                                     encoding="utf-8")


def test_select_gray_takes_a_3_4_star_and_b_neutral():
    a = pool()[lambda d: d.domain == "A"]
    b = pool()[lambda d: d.domain == "B"]
    assert list(select_gray(a, b).id) == ["tiki_1", "tiki_2", "res1_1"]


def test_select_gray_foody_uses_score_range():
    b = pd.DataFrame({"id": ["foody_1", "foody_2", "foody_3", "foody_4"], "domain": "B", "source": "foody",
                      "rating": [2.0, 3.0, 7.5, 8.2], "label": [0, 0, 2, 2]})
    assert list(select_gray(pool()[lambda d: d.domain == "A"].iloc[:0], b).id) == ["foody_2", "foody_3"]


def test_next_wave_never_reuses_batch_names(tmp_path):
    assert next_wave(tmp_path) == ""
    (tmp_path / "A_000.jsonl").write_text("")
    assert next_wave(tmp_path) == "2"
    (tmp_path / "B2_000.jsonl").write_text("")
    assert next_wave(tmp_path) == "3"


def test_export_batches_blind_and_sized(tmp_path):
    gray = select_gray(pool()[lambda d: d.domain == "A"], pool()[lambda d: d.domain == "B"])
    paths = export_batches(gray, tmp_path, batch_size=1)
    assert [p.name for p in paths] == ["A_000.jsonl", "A_001.jsonl", "B_000.jsonl"]
    item = json.loads(paths[0].read_text(encoding="utf-8").strip())
    assert item == {"id": "tiki_1", "text": "Sách rách bẩn quá"}   # no label leaked, whitespace collapsed


def test_validate_results_reports_problems(tmp_path):
    gray = select_gray(pool()[lambda d: d.domain == "A"], pool()[lambda d: d.domain == "B"])
    export_batches(gray, tmp_path / "batches", batch_size=2)
    write_results(tmp_path / "results", "A_000.jsonl",
                  [{"id": "tiki_1", "label": 0, "note": "chê"}, {"id": "tiki_9", "label": 3, "note": ""}])
    errs = validate_results(tmp_path / "batches", tmp_path / "results")
    joined = "\n".join(errs)
    assert "B_000.jsonl: missing result file" in joined
    assert "A_000.jsonl: missing ids ['tiki_2']" in joined
    assert "A_000.jsonl: unexpected ids ['tiki_9']" in joined
    assert "A_000.jsonl: invalid label" in joined
    assert "A_000.jsonl: empty note" in joined


def test_validate_and_merge_happy_path(tmp_path):
    gray = select_gray(pool()[lambda d: d.domain == "A"], pool()[lambda d: d.domain == "B"])
    export_batches(gray, tmp_path / "batches", batch_size=2)
    write_results(tmp_path / "results", "A_000.jsonl",
                  [{"id": "tiki_1", "label": 0, "note": "chê"}, {"id": "tiki_2", "label": 1, "note": "lẫn lộn"}])
    write_results(tmp_path / "results", "B_000.jsonl", [{"id": "res1_1", "label": 1, "note": "tạm"}])
    assert validate_results(tmp_path / "batches", tmp_path / "results") == []
    merged = merge_results(gray, tmp_path / "results")
    assert list(merged.columns) == ["id", "domain", "orig_label", "new_label", "note"]
    assert merged.set_index("id").loc["tiki_1", ["orig_label", "new_label"]].tolist() == [1, 0]


def test_apply_relabels_sets_label_and_source():
    df = pool().assign(label_source=["rating"] * 3 + ["dataset"] * 2)
    rel = pd.DataFrame({"id": ["tiki_1", "res1_1"], "new_label": [0, 1]})
    out = apply_relabels(df, rel)
    assert out.set_index("id").loc["tiki_1", "label"] == 0
    assert out.set_index("id").label_source.to_dict() == {
        "tiki_1": "content", "tiki_2": "rating", "tiki_3": "rating", "res1_1": "content", "res1_2": "dataset"}


def test_apply_relabels_drops_spam():
    df = pool().assign(label_source="rating")
    rel = pd.DataFrame({"id": ["tiki_1", "tiki_2", "tiki_3"], "new_label": [1, 0, 1],
                        "note": ["(spam) quảng cáo", "chê sách, spam nhiều", "bình thường"]})
    out = apply_relabels(df, rel)
    assert "tiki_1" not in set(out["id"])
    assert {"tiki_2", "tiki_3"} <= set(out["id"])


def test_apply_relabels_rejects_bad_label():
    with pytest.raises(ValueError):
        apply_relabels(pool().assign(label_source="rating"), pd.DataFrame({"id": ["tiki_1"], "new_label": [5]}))


def test_qa_against_audit_agreement():
    merged = pd.DataFrame({"id": ["a", "b", "c"], "domain": ["A"] * 3,
                           "orig_label": [1, 1, 2], "new_label": [1, 0, 2], "note": ["x"] * 3})
    audit = pd.DataFrame({"id": ["a", "b", "c", "z"], "label": [1, 1, 2, 0], "correct": [1, 0, 0, 1]})
    res = qa_against_audit(merged, audit)
    # a: audit says correct & kept -> agree; b: audit says wrong & changed -> agree; c: audit wrong but kept -> disagree
    assert res == {"n": 3, "agree": 2, "rate": round(2 / 3, 3)}


def test_validate_partial_skips_missing_files(tmp_path):
    gray = select_gray(pool()[lambda d: d.domain == "A"], pool()[lambda d: d.domain == "B"])
    export_batches(gray, tmp_path / "batches", batch_size=2)
    write_results(tmp_path / "results", "B_000.jsonl", [{"id": "res1_1", "label": 1, "note": "tạm"}])
    assert validate_results(tmp_path / "batches", tmp_path / "results", partial=True) == []
    assert validate_results(tmp_path / "batches", tmp_path / "results") == ["A_000.jsonl: missing result file"]
