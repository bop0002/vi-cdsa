import pandas as pd
from src.checks import run_checks


def make(prefix, n_per_class):
    rows = [{"key": f"{prefix} {c} {i}", "text_clean": f"{prefix} câu số {c} {i}", "label": c, "n_words": 4,
             "group_id": f"{prefix}-{c}-{i}"}
            for c in (0, 1, 2) for i in range(n_per_class)]
    return pd.DataFrame(rows)


def valid_frames():
    f = {}
    for d in "AB":
        f[f"{d}_train"] = make(f"{d}tr", 20)
        f[f"{d}_val"] = make(f"{d}va", 2)
        f[f"{d}_test"] = make(f"{d}te", 2)
        f[f"{d}_test_balanced"] = f[f"{d}_test"]
    return f


def test_valid_frames_pass():
    assert run_checks(valid_frames(), min_test=2, min_train=5) == []


def test_detects_split_overlap():
    f = valid_frames()
    f["A_test"] = pd.concat([f["A_test"], f["A_train"].head(1)])
    errs = run_checks(f, min_test=2, min_train=5)
    assert any("overlap train/test" in e for e in errs)


def test_detects_cross_domain_overlap_and_size_mismatch():
    f = valid_frames()
    f["B_train"] = pd.concat([f["B_train"], f["A_val"].head(1)])
    errs = run_checks(f, min_test=2, min_train=5)
    assert any("overlap A/B" in e for e in errs)
    assert any("train size" in e for e in errs)


def test_detects_small_class_and_auto_title():
    f = valid_frames()
    f["B_test"].loc[0, "text_clean"] = "hài lòng"
    errs = run_checks(f, min_test=3, min_train=5)
    assert any("B_test: class" in e for e in errs)
    assert any("auto-title" in e for e in errs)


def test_user_review_starting_with_title_phrase_is_not_flagged():
    f = valid_frames()
    f["B_test"].loc[0, "text_clean"] = "hài lòng, sách đẹp"
    errs = run_checks(f, min_test=3, min_train=5)
    assert not any("auto-title" in e for e in errs)


def test_detects_dominant_group_id():
    f = valid_frames()
    f["A_train"] = f["A_train"].assign(group_id="same")
    errs = run_checks(f, min_test=2, min_train=5)
    assert any("A: group_id share > 2%" in e for e in errs)


def test_equal_splits_detects_class_count_mismatch():
    f = valid_frames()
    f["B_test"] = make("Bte", 3)
    assert run_checks(f, min_test=2, min_train=5) == []
    assert any("test class counts differ" in e for e in run_checks(f, min_test=2, min_train=5,
                                                                   equal_splits=("train", "test")))
