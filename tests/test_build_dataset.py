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
