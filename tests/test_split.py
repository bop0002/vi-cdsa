import numpy as np
import pandas as pd
import pytest
from src.split import balanced_subset, group_split, match_train_size


def synthetic(n_groups=200, per_group=5, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_groups):
        for j in range(per_group):
            rows.append({"id": f"{g}_{j}", "group_id": f"g{g}",
                         "label": int(rng.choice([0, 1, 2], p=[0.1, 0.1, 0.8]))})
    return pd.DataFrame(rows)


def test_group_split_no_group_leak_and_ratios():
    df = group_split(synthetic())
    groups = {s: set(df[df.split == s].group_id) for s in ["train", "val", "test"]}
    assert not (groups["train"] & groups["val"]) and not (groups["train"] & groups["test"]) \
        and not (groups["val"] & groups["test"])
    frac = df.split.value_counts(normalize=True)
    assert abs(frac["train"] - 0.8) < 0.03 and abs(frac["test"] - 0.1) < 0.03


def test_group_split_deterministic():
    assert group_split(synthetic()).split.tolist() == group_split(synthetic()).split.tolist()


def test_match_train_size_copies_b_class_counts_and_only_touches_train():
    a = group_split(synthetic())
    b = pd.DataFrame({"split": ["train"] * 300 + ["val"] * 50 + ["test"] * 50,
                      "label": [0] * 60 + [1] * 30 + [2] * 210 + [2] * 100})
    out = match_train_size(a, b)
    assert out[out.split == "train"].label.value_counts().to_dict() == {0: 60, 1: 30, 2: 210}
    for s in ["val", "test"]:
        assert (out.split == s).sum() == (a.split == s).sum()


def test_match_train_size_fails_when_a_class_is_short():
    a = group_split(synthetic())
    b = pd.DataFrame({"split": ["train"] * 500, "label": [1] * 500})
    with pytest.raises(ValueError, match="lacks rows"):
        match_train_size(a, b)


def test_balanced_subset_equal_classes():
    t = pd.DataFrame({"id": range(100), "label": [0] * 10 + [1] * 20 + [2] * 70})
    out = balanced_subset(t)
    assert out.label.value_counts().tolist() == [10, 10, 10]
