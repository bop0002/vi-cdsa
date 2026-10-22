import pandas as pd

from src.build_dataset import write_audit_if_missing


def test_writes_audit_when_missing(tmp_path):
    p = tmp_path / "label_audit_A.csv"
    df = pd.DataFrame({"id": [1], "correct": [""]})
    assert write_audit_if_missing(df, p) is True
    assert p.exists()
    assert pd.read_csv(p, encoding="utf-8-sig").shape == (1, 2)


def test_does_not_overwrite_existing_audit(tmp_path, capsys):
    p = tmp_path / "label_audit_A.csv"
    p.write_text("id,correct\n1,y\n", encoding="utf-8-sig")
    before = p.read_bytes()
    assert write_audit_if_missing(pd.DataFrame({"id": [9], "correct": [""]}), p) is False
    assert p.read_bytes() == before
    assert "not overwritten" in capsys.readouterr().out


def test_load_relabels_absent_returns_none(tmp_path):
    from src.build_dataset import load_relabels
    assert load_relabels(tmp_path / "nope.csv") is None


def test_load_relabels_reads_id_and_new_label(tmp_path):
    from src.build_dataset import load_relabels
    p = tmp_path / "relabel_labels.csv"
    pd.DataFrame({"id": ["tiki_1"], "domain": ["A"], "orig_label": [1], "new_label": [0], "note": ["x"]}) \
        .to_csv(p, index=False, encoding="utf-8-sig")
    df = load_relabels(p)
    assert df.to_dict("records") == [{"id": "tiki_1", "new_label": 0, "note": "x"}]
