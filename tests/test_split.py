import numpy as np
import pandas as pd
import pytest
from src.split import add_extra_to_train, balanced_subset, frozen_group_split, group_split, match_train_size


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


def test_match_train_size_takes_min_per_class_and_only_touches_train():
    a = group_split(synthetic())
    have_a = a[a.split == "train"].label.value_counts()
    b = pd.DataFrame({"split": ["train"] * 2000 + ["val"] * 50,
                      "label": [0] * 60 + [1] * 1000 + [2] * 940 + [2] * 50, "orig_split": None})
    oa, ob = match_train_size(a, b)
    want = {0: 60, 1: int(have_a[1]), 2: int(have_a[2])}
    for out in (oa, ob):
        assert out[out.split == "train"].label.value_counts().sort_index().to_dict() == want
    assert (oa.split == "val").sum() == (a.split == "val").sum() and (ob.split == "val").sum() == 50


def test_match_train_size_cap_and_original_rows_first():
    b = pd.DataFrame({"split": "train", "label": 2, "id": range(10),
                      "orig_split": ["train"] * 4 + ["train_extra"] * 6})
    a = b.assign(orig_split=None)
    _, ob = match_train_size(a, b, cap={2: 5})
    assert len(ob) == 5 and (ob.orig_split == "train").sum() == 4


def test_add_extra_to_train_drops_held_out_groups():
    base = pd.DataFrame({"group_id": ["g1", "g2", "g3"], "split": ["train", "val", "test"]})
    extra = pd.DataFrame({"group_id": ["g1", "g2", "g3", "g4"]})
    out = add_extra_to_train(base, extra)
    assert out[len(base):].group_id.tolist() == ["g1", "g4"]
    assert (out[len(base):].split == "train").all()


def test_balanced_subset_equal_classes():
    t = pd.DataFrame({"id": range(100), "label": [0] * 10 + [1] * 20 + [2] * 70})
    out = balanced_subset(t)
    assert out.label.value_counts().tolist() == [10, 10, 10]


def test_frozen_group_split_uses_listed_groups(tmp_path):
    p = tmp_path / "held.csv"
    pd.DataFrame({"group_id": ["g1", "g2"], "split": ["val", "test"]}).to_csv(p, index=False)
    df = pd.DataFrame({"group_id": ["g1", "g2", "g3", "g1"], "label": [0, 1, 2, 2]})
    assert frozen_group_split(df, p)["split"].tolist() == ["val", "test", "train", "val"]


def test_frozen_group_split_falls_back_without_file(tmp_path):
    assert frozen_group_split(synthetic(), tmp_path / "nope.csv").split.tolist() == group_split(synthetic()).split.tolist()
